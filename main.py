import time
import numpy as np
import argparse
import copy
import os
import wandb
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

from configs import build_config
from utils import setup_seed
from log import get_logger
from model.model import XModel
from train_epoch import train_func
from test import test_func
from infer import infer_func
from loss.UR_DMU_loss import AD_Loss
from utils_data import get_datasets, get_dataloaders
from utils_log import log_training_status, evaluate_and_save_model
from utils_model import load_checkpoint, load_WSV_GST_model, save_model_weights


def train_epoch(
    model,
    train_nloader,
    train_aloader,
    test_loader,
    gt,
    criterion,
    criterion2,
    criterion3,
    optimizer,
    cfg,
    logger,
    logger_wandb,
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

            if epoch >= (0 if not cfg.fast else cfg.max_epoch) and (idx + 1) % 10 == 0:
                auc, ab_auc = test_func(test_loader, model, gt, cfg.dataset, cfg.test_bs)
                if auc > best_auc:
                    best_auc = auc
                    auc_ab_auc = ab_auc
                    torch.save(model.state_dict(), os.path.join(cfg.save_dir, f"{cfg.model_name}_current.pkl"))

                log_training_status(epoch, idx, loss1, loss2, loss3, auc, ab_auc, best_auc, cfg, logger)

        # Evaluate and save best model
        auc, ab_auc = evaluate_and_save_model(
            epoch, model, test_loader, test_func, optimizer, gt, cfg, best_auc, logger
        )
        if auc >= best_auc:
            best_auc, auc_ab_auc = auc, ab_auc

    return best_auc, auc_ab_auc

def get_optimizer(cfg, model):
    XEncoder_params = list(map(id, model.self_attention.parameters()))
    main_train_params_filter = filter(lambda p: id(p) not in XEncoder_params, model.parameters())
    optimizer = optim.Adam(
        [
            {
                "params": model.self_attention.parameters(),
                "lr": 1e-7,
            },
            {
                "params": main_train_params_filter,
                "lr": cfg.lr,
            },
        ]
    )
    return optimizer

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
        load_WSV_GST_model(model, cfg, logger)
        criterion, criterion2, criterion3 = torch.nn.BCELoss(), torch.nn.KLDivLoss(reduction="batchmean"), AD_Loss()
        
        optimizer = get_optimizer(cfg, model)
        # logger.info("Model:{}\n".format(model))
        logger.info("Optimizer:{}\n".format(optimizer))
        param = sum(p.numel() for p in model.parameters())
        logger.info("total params:{:.4f}M".format(param / (1000**2)))

        best_auc, auc_ab_auc = train_epoch(
            model,
            train_nloader,
            train_aloader,
            test_loader,
            gt,
            criterion,
            criterion2,
            criterion3,
            optimizer,
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


def initialize_wandb(args, cfg):
    """Initialize Weights and Biases logging."""
    if args.disable_wandb:
        return None
    name = f"{args.dataset}_{args.version}_{cfg.lr}_{cfg.train_bs}_Mem{cfg.a_nums}_{cfg.n_nums}"
    logger_wandb = wandb.init(
        project=f"WSV-GST_{args.dataset}(clip+i3d+audio)", name=name, group=f"epoch-{args.version}(clip-pel-ur)"
    )
    logger_wandb.config.update(args)
    logger_wandb.config.update(cfg.__dict__, allow_val_change=True)
    return logger_wandb


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

    if cfg.dataset != "xd-violence":
        print(f"{cfg.dataset} is not supported. XD-Violence is only supported for RGB+audio.")
        exit()

    train_model(cfg, args)
