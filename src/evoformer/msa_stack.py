from torch import layer_norm
import torch
import torch.nn as nn
from src.attention.mha import MultiHeadAttention
from torch import Tensor


class MSARowAttentionWithPairBias(nn.Module):
    """
    Implements Algorithm.
    """
    def __init__(self, c_m: int, c_z: int, c: int=32, N_head: int=8):
        """
        Initializes MSARowAttentionWithPairBias.

        Args:
            c_m (int): Embedding dimension of the msa representation.
            c_z (int): Embedding dimension of the pair representation.
            c (int, optional): Embedding dimension for multi-head attention. Defaults to 32.
            N_head (int, optional): Number of heads for multi-head attention. Defaults to 8.
        """
        super().__init__()

        self.layer_norm_m = nn.LayerNorm(c_m)
        self.layer_norm_z = nn.LayerNorm(c_z)
        self.linear_z = nn.Linear(c_z, N_head, bias=False)
        self.mha = MultiHeadAttention(c_m, c, N_head, attn_dim=-2, gated=True)

    def forward(self, m: Tensor, z: Tensor) -> Tensor:
        """
        Implements the forward pass according to Algorithm 7.

        Args:
            m (torch.tensor): MSA representation of shape (*, N_seq, N_res, c_m).
            z (torch.tensor): Pair representation of shape (*, N_res, N_res, c_z).

        Returns:
            torch.tensor: Output tensor of the same shape as m.
        """

        m = self.layer_norm_m(m)
        z = self.layer_norm_z(z)

        b = self.linear_z(z)
        b = b.moveaxis(-1, -3)
        out = self.mha(m, bias=b)

        return out


class MSAColumnAttention(nn.Module):
    """
    Implements Algorithm 8.
    """
    def __init__(self, c_m: int, c: int=32, N_head: int=8):
        """
        Initializes MSAColumnAttention.

        Args:
            c_m (int): Embedding dimension of the MSA representation.
            c (int, optional): Embedding dimension for multi-head attention. Defaults to 32.
            N_head (int, optional): Number of heads for multi-head attention. Defaults to 8.
        """
        super().__init__()

        self.layer_norm_m = nn.LayerNorm(c_m)
        self.mha = MultiHeadAttention(c_m, c, N_head, attn_dim=-3, gated=True)

    def forward(self, m: Tensor) -> Tensor:
        """
        Implements the forward pass according to algorithm Algorithm 8.

        Args:
            m (torch.tensor): MSA representation of shape (N_seq, N_res, c_m).

        Returns:
            torch.tensor: Output tensor of the same shape as m.
        """

        m = self.layer_norm_m(m)
        out = self.mha(m)

        return out


class MSATransition(nn.Module):
    """
    Implements Algorithm 9.
    """
    def __init__(self, c_m: int, n: int=4):
        """
        Initializes MSATransition.

        Args:
            c_m (int): Embedding dimension of the MSA representation.
            n (int, optional): Factor for the number of channels in the intermediate dimension. 
             Defaults to 4.
        """
        super().__init__()

        self.layer_norm = nn.LayerNorm(c_m)
        self.linear_1 = nn.Linear(c_m, n*c_m)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(n*c_m, c_m)

    def forward(self, m: Tensor) -> Tensor:
        """
        Implements the forward pass for Algorithm 9.

        Args:
            m (torch.tensor): MSA feat of shape (*, N_seq, N_seq, c_m).

        Returns:
            torch.tensor: Output tensor of the same shape as m.
        """

        m = self.layer_norm(m)
        m = self.linear_1(m)
        m = self.relu(m)
        m = self.linear_2(m)

        return m


class OuterProductMean(nn.Module):
    """
    Implements Algorithm 10.
    """
    def __init__(self, c_m: int, c_z: int, c: int=32):
        """
        Initializes OuterProductMean.

        Args:
            c_m (int): Embedding dimension of the MSA representation.
            c_z (int): Embedding dimension of the pair representation. 
            c (int, optional): Embedding dimension of a and b from Algorithm 10. 
                Defaults to 32.
        """
        super().__init__()

        self.layer_norm = nn.LayerNorm(c_m)
        self.linear_1 = nn.Linear(c_m, c)
        self.linear_2 = nn.Linear(c_m, c)
        self.linear_out = nn.Linear(c*c, c_z)

    def forward(self, m: Tensor) -> Tensor:
        """
        Forward pass for Algorithm 10.

        Args:
            m (torch.tensor): MSA feat of shape (*, N_seq, N_res, c_m).

        Returns:
            torch.tensor: Output tensor of shape (*, N_res, N_res, c_z).
        """
        N_seq = m.shape[-3]

        m = self.layer_norm(m)
        a = self.linear_1(m)
        b = self.linear_2(m)
        o = torch.einsum('...sic,...sjd->...ijcd', a, b)
        o = torch.flatten(o, start_dim=-2)
        z = self.linear_out(o) / N_seq

        return z