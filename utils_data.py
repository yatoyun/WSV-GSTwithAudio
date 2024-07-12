from torch.utils.data import DataLoader

from dataset import UCFDataset, XDataset, SHDataset

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
    train_nloader = DataLoader(
        train_normal_data,
        batch_size=cfg.train_bs,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=True,
        drop_last=True,
    )
    train_aloader = DataLoader(
        train_anomaly_data,
        batch_size=cfg.train_bs,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(test_data, batch_size=cfg.test_bs, shuffle=False, num_workers=cfg.workers, pin_memory=True)
    return train_nloader, train_aloader, test_loader