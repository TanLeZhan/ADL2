import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)
def fold_to_dataloader_tensor(train_x, test_x, train_y, test_y, batch_size=64, device=device):
    # Convert all to DataFrames to simplify null checks and conversions
    train_x = pd.DataFrame(train_x).apply(pd.to_numeric, errors='coerce')
    test_x = pd.DataFrame(test_x).apply(pd.to_numeric, errors='coerce')
    train_y = pd.DataFrame(train_y).squeeze().apply(pd.to_numeric, errors='coerce')
    test_y = pd.DataFrame(test_y).squeeze().apply(pd.to_numeric, errors='coerce')

    # Check for any NaNs after conversion
    for name, df in zip(['train_x', 'test_x', 'train_y', 'test_y'], [train_x, test_x, train_y, test_y]):
        if df.isnull().values.any():
            bad_cols = df.columns[df.isnull().any()] if hasattr(df, 'columns') else []
            raise ValueError(f"❌ NaNs detected in {name} after conversion in columns: {list(bad_cols)}")

    # Create PyTorch datasets and loaders
    train_dataset = TensorDataset(
                                torch.tensor(train_x.values, dtype=torch.float32).to(device),
                                torch.tensor(train_y.values, dtype=torch.float32).view(-1, 1).to(device)
                            )
    val_dataset = TensorDataset(
                            torch.tensor(test_x.values, dtype=torch.float32).to(device),
                            torch.tensor(test_y.values, dtype=torch.float32).view(-1, 1).to(device)
                        )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=True, drop_last=True)

    return train_loader, val_loader

def get_feature_count(loader):
    """returns the number of features in the dataset"""
    return next(iter(loader))[0].shape[1]


def init_weights(model):
    if isinstance(model, nn.Linear):  # Apply only to linear layers
        # He initialization (recommended for ReLU activations)
        # print("Initializing weights using kaiming")
        nn.init.kaiming_normal_(model.weight, mode='fan_in', nonlinearity='relu')
        
        # Bias initialization (zero initialization is fine)
        if model.bias is not None:
            nn.init.zeros_(model.bias)
            
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        probs = probs.clamp(min=1e-6, max=1 - 1e-6)  # avoid log(0)

        targets = targets.type_as(inputs)

        # focal loss components
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        modulating_factor = (1 - p_t) ** self.gamma

        loss = alpha_factor * modulating_factor * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
def criterion_mapping(criterion_choice:str, pos_weight:float=None, alpha:float=None, gamma:float=None):
    """
    Feel free to add any custom loss functions here.
    returns function for criterion
    """
    if criterion_choice == "FocalLoss":
        return FocalLoss(alpha = alpha, gamma = gamma)
    elif criterion_choice == "BCEWithLogitsLoss":
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight])) if pos_weight else nn.BCEWithLogitsLoss()
    return nn.BCEWithLogitsLoss() 
