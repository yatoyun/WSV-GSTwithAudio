import time
import numpy as np
import argparse
import copy
import os
import sys
import wandb
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import build_config
from utils import setup_seed
from log import get_logger
from dataset import UCFDataset, XDataset, SHDataset
from model.model import XModel
from pesudo.GANmodel import Autoencoder
from pesudo.test import test_func
from infer import infer_func
from loss.UR_DMU_loss import AD_Loss


def load_checkpoint(model, ckpt_path, logger):
    """Load model weights from a checkpoint file."""
    if os.path.isfile(ckpt_path):
        logger.info(f"Loading pretrained checkpoint from {ckpt_path}.")
        weight_dict = torch.load(ckpt_path)
        model_dict = model.state_dict()
        for name, param in weight_dict.items():
            if "module" in name:
                name = ".".join(name.split(".")[1:])
            if name in model_dict and param.size() == model_dict[name].size():
                model_dict[name].copy_(param)
            else:
                logger.info(f"{name} size mismatch: load {param.size()} given {model_dict[name].size()}")
    else:
        logger.info("Pretrained checkpoint file not found.")

def pad_tensor(tensor, max_seqlen):
    batch_size, num_frames, feature_dim = tensor.size()
    padding_length = max_seqlen - num_frames
    if padding_length == 0:
        return tensor
    padded_tensor = torch.cat([tensor, torch.zeros(batch_size, padding_length, feature_dim)], dim=1)
    return padded_tensor

def train_epoch(
    WSV_model,
    model,
    train_nloader,
    train_aloader,
    test_loader,
    gt,
    criterion,
    criterion2,
    criterion3,
    scheduler,
    cfg,
    logger,
    logger_wandb,
):
    """Train model for one epoch."""
    best_auc = 0.0
    auc_ab_auc = 0.0
    min_loss = 1000000

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    acc1, acc2 = test_func(WSV_model, model, test_loader, criterion, gt, cfg, logger)
    logger.info(f'Initial test |  Acc1: {acc1:.4f}, Acc2: {acc2:.4f}')
    for epoch in range(cfg.max_epoch):
        for i, (v_input, clip_input, t_input, a_input, label, multi_label) in enumerate(train_nloader):
            with torch.no_grad():
                WSV_model.eval()
                real_data = torch.zeros(0).cuda()
                total_seq_len = torch.sum(torch.max(torch.abs(v_input), dim=2)[0] > 0, 1)
                ex_seq_len = 0
                while total_seq_len[0] > 0:
                    seq_len = torch.Tensor([min(10000, total_seq_len[0])]).to(torch.int64)
                    total_seq_len -= seq_len
                    v_in = v_input[:, ex_seq_len:ex_seq_len+torch.max(seq_len), :]
                    clip_in = clip_input[:, ex_seq_len:ex_seq_len+torch.max(seq_len), :]
                    a_in = a_input[:, ex_seq_len:ex_seq_len+torch.max(seq_len), :]
                    ex_seq_len += seq_len[0]
                    
                    clip_in = pad_tensor(clip_in, torch.max(seq_len))
                    
                    v_in = v_in.float().cuda(non_blocking=True)
                    clip_in = clip_in.float().cuda(non_blocking=True)
                    a_in = a_in.float().cuda(non_blocking=True)

                    # print(v_in.shape, clip_in.shape, a_in.shape, seq_len)
                    logits, x_k = WSV_model(v_in, clip_in, a_in, seq_len)
                    real_data = x_k["x"]
                    del v_in, clip_in, a_in, x_k
                    real_data = torch.cat((real_data, real_data), 0)
            with torch.set_grad_enabled(True):
                model.train()
                # 本物のデータ
                inputs = real_data.to('cuda')
                inputs = inputs.permute(0, 2, 1)
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        if min_loss > loss.item():
            min_loss = loss.item()
            torch.save(model.state_dict(), cfg.ckpt_path)
        
        acc1, acc2 = test_func(WSV_model, model, test_loader, criterion, gt, cfg, logger)
        logger.info(f'Epoch [{epoch+1}/{cfg.max_epoch}] Loss: {loss.item():.4f} |  Acc1: {acc1:.4f}, Acc2: {acc2:.4f}')

        # scheduler.step()
        # Evaluate and save best model

    return best_auc, auc_ab_auc


def log_training_status(epoch, idx, loss1, loss2, loss3, auc, ab_auc, best_auc, logger):
    """Log training status."""
    logger.info(
        f"[Epoch:{epoch + 1}/{cfg.max_epoch}, Batch:{idx}]: loss1:{loss1:.4f} loss2:{loss2:.4f} loss3:{loss3:.4f} | BestAP:{best_auc:.4f} AP:{auc:.4f} Anomaly AUC:{ab_auc:.4f}"
    )

def load_specific_weights(model, checkpoint_path, prefix='self_attention'):
    checkpoint = torch.load(checkpoint_path)
    model_dict = model.state_dict()
    filtered_dict = {k: v for k, v in checkpoint.items() if k.startswith(prefix)}

    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)

def load_WSV_GST_model(model, cfg, logger):
    try:
        load_specific_weights(model, cfg.WSV_GST_model_path, prefix='self_attention')
        logger.info(f"Success: Load WSV-GST model")
    except Exception as e:
        logger.info(f"Fail: Load WSV-GST model")
        logger.info(f"Error: {e}")
        raise e


def train_model(cfg, args):
    """Train the model based on the provided configuration and arguments."""
    logger = get_logger(cfg.logs_dir)
    setup_seed(cfg.seed)
    logger.info(f"Config: {cfg.__dict__}")

    train_normal_data, train_anomaly_data, test_data = get_datasets(cfg)
    train_nloader, train_aloader, test_loader = get_dataloaders(cfg, train_normal_data, train_anomaly_data, test_data)

    WSV_model = XModel(cfg).to(torch.device("cuda"))
    model = Autoencoder(300, 128).to('cuda')
    gt = np.load(cfg.gt)

    if args.mode == "train":
        logger_wandb = initialize_wandb(args, cfg)
        load_WSV_GST_model(WSV_model, cfg, logger)
        criterion, criterion2, criterion3 = torch.nn.MSELoss(), torch.nn.KLDivLoss(reduction="batchmean"), AD_Loss()
        # optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)
        scheduler = None #torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        # logger.info("Model:{}\n".format(model))
        param = sum(p.numel() for p in model.parameters())
        logger.info("total params:{:.4f}M".format(param / (1000**2)))

        best_auc, auc_ab_auc = train_epoch(
            WSV_model,
            model,
            train_nloader,
            train_aloader,
            test_loader,
            gt,
            criterion,
            criterion2,
            criterion3,
            scheduler,
            cfg,
            logger,
            logger_wandb,
        )

        # Save final model weights
        save_model_weights(cfg, best_auc, logger)

    elif args.mode == "infer":
        infer_mode(cfg, model, logger, test_loader, gt)

    else:
        raise RuntimeError("Invalid mode!")


def get_datasets(cfg):
    """Retrieve the datasets based on the configuration."""
    if cfg.dataset == "ucf-crime":
        train_normal_data = UCFDataset(cfg, test_mode=False)
        train_anomaly_data = UCFDataset(cfg, test_mode=False, is_abnormal=True)
        test_data = UCFDataset(cfg, test_mode=True)
    elif cfg.dataset == "xd-violence":
        train_normal_data = XDataset(cfg, test_mode=False)
        train_anomaly_data = XDataset(cfg, test_mode=False, is_abnormal=True)
        test_data = XDataset(cfg, test_mode=True)
    elif cfg.dataset == "shanghaiTech":
        train_normal_data = SHDataset(cfg, test_mode=False)
        train_anomaly_data = SHDataset(cfg, test_mode=False, is_abnormal=True)
        test_data = SHDataset(cfg, test_mode=True)
    else:
        raise RuntimeError(f"Dataset {cfg.dataset} is not supported!")
    return train_normal_data, train_anomaly_data, test_data


def get_dataloaders(cfg, train_normal_data, train_anomaly_data, test_data):
    """Create dataloaders for training and testing."""
    train_nloader = DataLoader(
        train_normal_data,
        batch_size=cfg.train_bs,
        num_workers=cfg.workers,
        pin_memory=True,
        drop_last=True,
    )
    # train_aloader = DataLoader(
    #     train_anomaly_data,
    #     batch_size=cfg.train_bs,
    #     shuffle=True,
    #     num_workers=cfg.workers,
    #     pin_memory=True,
    #     drop_last=True,
    # )
    test_loader = DataLoader(test_data, batch_size=cfg.test_bs, shuffle=False, num_workers=cfg.workers, pin_memory=True)
    return train_nloader, None, test_loader


def initialize_wandb(args, cfg):
    """Initialize Weights and Biases logging."""
    return None
    if args.disable_wandb:
        return None
    name = f"{args.dataset}_{args.version}_{cfg.lr}_{cfg.train_bs}_Mem{cfg.a_nums}_{cfg.n_nums}"
    logger_wandb = wandb.init(
        project=f"WSV-GST_{args.dataset}(clip+i3d+audio)", name=name, group=f"epoch-{args.version}(clip-pel-ur)"
    )
    logger_wandb.config.update(args)
    logger_wandb.config.update(cfg.__dict__, allow_val_change=True)
    return logger_wandb


def save_model_weights(cfg, best_auc, logger):
    """Save the model weights to disk."""
    current_best_model_path = os.path.join(cfg.save_dir, f"{cfg.model_name}_current.pkl")
    save_path = os.path.join(cfg.save_dir, f"{cfg.model_name}_{str(round(best_auc, 4)).split('.')[1]}.pkl")

    os.system(f"cp {current_best_model_path} {save_path}")
    logger.info(f"Model saved to {save_path}")


def infer_mode(cfg, model, logger, test_loader, gt):
    """Run the model in inference mode."""
    logger.info("Inference Mode")
    if cfg.ckpt_path is None:
        logger.info("Checkpoint path is None!")
        raise ValueError("Checkpoint path is None!")
    load_checkpoint(model, cfg.ckpt_path, logger)
    infer_func(model, test_loader, gt, logger, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WeaklySupAnoDet")
    parser.add_argument("--dataset", default="xd", help="Anomaly video dataset")
    parser.add_argument("--mode", default="train", help="Model status: (train or infer)")
    parser.add_argument("--version", default="original", help="Change log path name")
    parser.add_argument("--lamda", default=None, type=float, help="Lambda value")
    parser.add_argument("--alpha", default=None, type=float, help="Alpha value")
    parser.add_argument("--fast", action="store_true", help="Fast mode")
    parser.add_argument("-dw", "--disable_wandb", action="store_true", help="Disable Weights and Biases logging")

    args = parser.parse_args()
    cfg = build_config(args.dataset)

    # Update cfg
    cfg.lamda = args.lamda if args.lamda is not None else cfg.lamda
    cfg.alpha = args.alpha if args.alpha is not None else cfg.alpha
    cfg.fast = args.fast
    cfg.train_bs = 1 #cfg.test_bs
    savepath = f"./logs/{args.dataset}_{args.version}_{cfg.lr}_{cfg.train_bs}"
    os.makedirs(savepath, exist_ok=True)

    train_model(cfg, args)
