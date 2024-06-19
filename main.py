import time
import numpy as np
import argparse
import copy
import os
import wandb
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from configs import build_config
from utils import setup_seed
from log import get_logger
from dataset import UCFDataset, XDataset, SHDataset
from model.model import XModel
from train_epoch import train_func
from test import test_func
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


def train_epoch(
    model, train_nloader, train_aloader, test_loader, gt, criterion, criterion2, criterion3, optimizer, cfg, logger, log_writer, logger_wandb
):
    """Train model for one epoch."""
    best_auc = 0.0
    auc_ab_auc = 0.0

    for epoch in range(cfg.max_epoch):
        for idx, (n_input, a_input) in enumerate(zip(train_nloader, train_aloader)):
            loss1, loss2, loss3 = train_func(
                n_input,
                a_input,
                model,
                optimizer,
                criterion,
                criterion2,
                criterion3,
                logger_wandb,
                cfg.lamda,
                cfg.alpha,
                cfg.margin,
            )
            log_writer.add_scalar("loss", loss1, epoch)

            if epoch >= (1 if not cfg.fast else cfg.max_epoch) and (idx + 1) % 10 == 0:
                auc, ab_auc = test_func(test_loader, model, gt, cfg.dataset, cfg.test_bs)
                if auc > best_auc:
                    best_auc = auc
                    auc_ab_auc = ab_auc
                    torch.save(model.state_dict(), os.path.join(cfg.save_dir, f"{cfg.model_name}_current.pkl"))

                log_writer.add_scalar("AP", auc, epoch)
                log_training_status(epoch, idx, loss1, loss2, loss3, auc, ab_auc, best_auc, logger)

        # Evaluate and save best model
        auc, ab_auc = evaluate_and_save_model(epoch, model, test_loader, gt, cfg, best_auc, logger, log_writer)
        if auc >= best_auc:
            best_auc, auc_ab_auc = auc, ab_auc

    return best_auc, auc_ab_auc


def log_training_status(epoch, idx, loss1, loss2, loss3, auc, ab_auc, best_auc, logger):
    """Log training status."""
    logger.info(f"[Epoch:{epoch + 1}/{cfg.max_epoch}, Batch:{idx}]: loss1:{loss1:.4f} loss2:{loss2:.4f} loss3:{loss3:.4f} | BestAP:{best_auc:.4f} AP:{auc:.4f} Anomaly AUC:{ab_auc:.4f}")

def evaluate_and_save_model(epoch, model, test_loader, gt, cfg, best_auc, logger, log_writer):
    """Evaluate model and save the best model weights."""
    auc, ab_auc = test_func(test_loader, model, gt, cfg.dataset, cfg.test_bs)
    if auc > best_auc:
        torch.save(model.state_dict(), os.path.join(cfg.save_dir, f"{cfg.model_name}_current.pkl"))
    log_writer.add_scalar("AP", auc, epoch)
    logger.info(f"[Epoch:{epoch + 1}/{cfg.max_epoch}]: BestAP:{best_auc:.4f} AP:{auc:.4f} Anomaly AUC:{ab_auc:.4f}")
    return auc, ab_auc

def train_model(cfg, args):
    """Train the model based on the provided configuration and arguments."""
    logger = get_logger(cfg.logs_dir)
    setup_seed(cfg.seed)
    logger.info(f"Config: {cfg.__dict__}")

    train_normal_data, train_anomaly_data, test_data = get_datasets(cfg)
    train_nloader, train_aloader, test_loader = get_dataloaders(cfg, train_normal_data, train_anomaly_data, test_data)

    model = XModel(cfg).to(torch.device("cuda"))
    gt = np.load(cfg.gt)

    if args.mode == "train":
        logger_wandb = initialize_wandb(args, cfg)
        criterion, criterion2, criterion3 = torch.nn.BCELoss(), torch.nn.KLDivLoss(reduction="batchmean"), AD_Loss()
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=0.005)
        
        logger.info("Optimizer:{}\n".format(optimizer))
        param = sum(p.numel() for p in model.parameters())
        logger.info("total params:{:.4f}M".format(param / (1000**2)))
    
        best_auc, auc_ab_auc = train_epoch(
            model, train_nloader, train_aloader, test_loader, gt, criterion, criterion2, criterion3, optimizer, cfg, logger, log_writer, logger_wandb
        )

        # Save final model weights
        save_model_weights(model, cfg, best_auc, logger)

    elif args.mode == "infer":
        infer_mode(cfg, model, logger, test_loader, gt)

    else:
        raise RuntimeError("Invalid mode!")


def get_datasets(cfg):
    """Retrieve the datasets based on the configuration."""
    if cfg.dataset == "ucf-crime":
        train_normal_data = UCFDataset(cfg, test_mode=False, pre_process=True)
        train_anomaly_data = UCFDataset(cfg, test_mode=False, is_abnormal=True, pre_process=True)
        test_data = UCFDataset(cfg, test_mode=True)
    elif cfg.dataset == "xd-violence":
        train_normal_data = XDataset(cfg, test_mode=False, pre_process=True)
        train_anomaly_data = XDataset(cfg, test_mode=False, is_abnormal=True, pre_process=True)
        test_data = XDataset(cfg, test_mode=True)
    elif cfg.dataset == "shanghaiTech":
        train_normal_data = SHDataset(cfg, test_mode=False, pre_process=True)
        train_anomaly_data = SHDataset(cfg, test_mode=False, is_abnormal=True, pre_process=True)
        test_data = SHDataset(cfg, test_mode=True)
    else:
        raise RuntimeError(f"Dataset {cfg.dataset} is not supported!")
    return train_normal_data, train_anomaly_data, test_data


def get_dataloaders(cfg, train_normal_data, train_anomaly_data, test_data):
    """Create dataloaders for training and testing."""
    train_nloader = DataLoader(train_normal_data, batch_size=cfg.train_bs, shuffle=True, num_workers=cfg.workers, pin_memory=True, drop_last=True)
    train_aloader = DataLoader(train_anomaly_data, batch_size=cfg.train_bs, shuffle=True, num_workers=cfg.workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=cfg.test_bs, shuffle=False, num_workers=cfg.workers, pin_memory=True)
    return train_nloader, train_aloader, test_loader


def initialize_wandb(args, cfg):
    """Initialize Weights and Biases logging."""
    if args.disable_wandb:
        return None
    name = f"{args.dataset}_{args.version}_{cfg.lr}_{cfg.train_bs}_Mem{cfg.a_nums}_{cfg.n_nums}"
    logger_wandb = wandb.init(project=f"WSV-GST_{args.dataset}(clip+i3d+audio)", name=name, group=f"epoch-{args.version}(clip-pel-ur)")
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

    savepath = f"./logs/{args.dataset}_{args.version}_{cfg.lr}_{cfg.train_bs}"
    os.makedirs(savepath, exist_ok=True)
    log_writer = SummaryWriter(savepath)

    if cfg.dataset != "xd-violence":
        print(f"{cfg.dataset} is not supported. XD-Violence is only supported for RGB+audio.")
        exit()

    train_model(cfg, args)
