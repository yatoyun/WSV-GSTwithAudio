import os
import torch

def log_training_status(epoch, idx, loss1, loss2, loss3, auc, ab_auc, best_auc, cfg, logger):
    """Log training status."""
    logger.info(
        f"[Epoch:{epoch + 1}/{cfg.max_epoch}, Batch:{idx}]: loss1:{loss1:.4f} loss2:{loss2:.4f} loss3:{loss3:.4f} | BestAP:{best_auc:.4f} AP:{auc:.4f} Anomaly AUC:{ab_auc:.4f}"
    )


def evaluate_and_save_model(epoch, model, test_loader, test_func, optimizer, gt, cfg, best_auc, logger):
    """Evaluate model and save the best model weights."""
    auc, ab_auc = test_func(test_loader, model, gt, cfg.dataset, cfg.test_bs)
    if auc > best_auc:
        torch.save(model.state_dict(), os.path.join(cfg.save_dir, f"{cfg.model_name}_current.pkl"))
    
    lr1 = optimizer.param_groups[0]["lr"]
    # lr2 = optimizer.param_groups[1]["lr"]
    logger.info(f"[Epoch:{epoch + 1}/{cfg.max_epoch}]: lr1:{lr1:.4f} | BestAP:{best_auc:.4f} AP:{auc:.4f} Anomaly AUC:{ab_auc:.4f}")
    return auc, ab_auc