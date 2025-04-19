import torch
from torch.utils.data import TensorDataset, DataLoader


def fold_to_dataloader_tensor(train_x, test_x, train_y, test_y, batch_size=64, device=device):
    train_dataset = TensorDataset(
        torch.tensor(train_x.values,dtype=torch.float32).to(device), 
        torch.tensor(train_y.values,dtype=torch.float32).to(device))
    val_dataset = TensorDataset(
        torch.tensor(test_x.values,dtype=torch.float32).to(device), 
        torch.tensor(test_y.values,dtype=torch.float32).to(device))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=True, drop_last=True)
    return train_loader, val_loader 

def get_feature_count(loader):
    """returns the number of features in the dataset"""
    return next(iter(loader))[0].shape[1]