import torch
import torch.nn as nn
from torch import Tensor


class RecyclingEmbedder(nn.Module):
    """
    Implements Algorithm 32.
    """

    def __init__(self, c_m: int, c_z: int):
        """
        Initializes the RecyclingEmbedder.

        Args:
            c_m (int): Embedding dimension of the MSA representation.
            c_z (int): Embedding dimension of the pair representation.
        """
        super().__init__()
        self.bin_start = 3.25
        self.bin_end = 20.75
        self.bin_count = 15

        self.layer_norm_m = nn.LayerNorm(c_m)
        self.layer_norm_z = nn.LayerNorm(c_z)
        self.linear = nn.Linear(self.bin_count, c_z)

    def forward(self, m_prev: Tensor, z_prev: Tensor, x_prev: Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward pass for Algorithm 32.

        Args:
            m_prev (torch.tensor): MSA representation of previous iteration, shape (*, N_seq, N_res, c_m).
            z_prev (torch.tensor): Pair representation of previous iteration, shape (*, N_res, N_res, c_z).
            x_prev (torch.tensor): Pseudo-beta positions from the previous iterations of 
                shape (*, N_res, 3). These are the positions of the C-beta atoms from the 
                last prediction (in Angstrom), or of C-alpha for glycin.
            

        Returns:
            tuple: A tuple consisting of m_out of shape (*, N_res, c_m) and z_out 
                of shape (*, N_res, N_res, c_z).
        """

        d = torch.linalg.vector_norm(x_prev.unsqueeze(-2) - x_prev.unsqueeze(-3), dim=-1)

        bins_lower = torch.linspace(self.bin_start, self.bin_end, self.bin_count, device=x_prev.device)
        bins_upper = torch.cat((bins_lower[1:], torch.tensor([1e8], device=x_prev.device)))

        d = d.unsqueeze(-1)
        d = ((d>bins_lower) * (d<bins_upper)).type(x_prev.dtype)
        d = self.linear(d)

        z_out = d + self.layer_norm_z(z_prev)
        m_out = self.layer_norm_m(m_prev[..., 0, :, :])

        return m_out, z_out