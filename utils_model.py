import os
import torch

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

def save_model_weights(cfg, best_auc, logger):
    """Save the model weights to disk."""
    current_best_model_path = os.path.join(cfg.save_dir, f"{cfg.model_name}_current.pkl")
    save_path = os.path.join(cfg.save_dir, f"{cfg.model_name}_{str(round(best_auc, 4)).split('.')[1]}.pkl")

    os.system(f"cp {current_best_model_path} {save_path}")
    logger.info(f"Model saved to {save_path}")