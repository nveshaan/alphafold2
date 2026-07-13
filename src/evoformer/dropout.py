import torch
import torch.nn as nn
from torch import Tensor

class SharedDropout(nn.Module):
    """
    A module for dropout, that is shared along one dimension,
    i.e. for dropping out whole rows or columns.
    """
    def __init__(self, shared_dim: int, p: float):
        super().__init__()

        self.dropout = nn.Dropout(p)
        self.shared_dim = shared_dim

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for shared dropout. The dropout mask is broadcasted along
        the shared dimension.

        Args:
            x (Tensor): Input tensor of arbitrary shape.

        Returns:
            Tensor: Output tensor of the same shape as x.
        """

        mask_shape = list(x.shape)
        mask_shape[self.shared_dim] = 1
        mask = torch.ones(mask_shape, device=x.device)
        mask = self.dropout(mask)

        out = x * mask

        return out

class DropoutRowwise(SharedDropout):
    def __init__(self, p: float):
        super().__init__(shared_dim=-2, p=p)

class DropoutColumnwise(SharedDropout):
    def __init__(self, p: float):
        super().__init__(shared_dim=-3, p=p)