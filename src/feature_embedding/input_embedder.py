import torch
import torch.nn as nn
from torch import Tensor


class InputEmbedder(nn.Module):
    """
    Implements Algorithms 3 and 4.
    """

    def __init__(self, c_m: int, c_z: int, tf_dim: int, msa_feat_dim: int=49, vbins: int=32):
        """
        Initializes the InputEmbedder.

        Args:
            c_m (int): Embedding dimension of the MSA representation.
            c_z (int): Embedding dimension of the pair representation.
            tf_dim (int): Embedding dimension of target_feat.
            msa_feat_dim (int, optional): Embedding dimension of the MSA feature. 
                Defaults to 49.
            vbins (int, optional): Determines the bins for relpos as 
                (-vbins, -vbins+1,...,vbins). Defaults to 32.
        """
        super().__init__()
        self.tf_dim = tf_dim
        self.vbins = vbins

        self.linear_tf_z_i = nn.Linear(tf_dim, c_z)
        self.linear_tf_z_j = nn.Linear(tf_dim, c_z)
        self.linear_tf_m = nn.Linear(tf_dim, c_m)
        self.linear_msa_m = nn.Linear(msa_feat_dim, c_m)
        self.linear_relpos = nn.Linear(2*vbins+1, c_z)

    def relpos(self, residue_index: Tensor) -> Tensor:
        """
        Implements Algorithm 4.

        Args:
            residue_index (torch.tensor): Index of the residue in the original amino
                acid sequence. In this context, this is simply [0,... N_res-1].

        Returns:
            tensor: relpos embedding
        """

        dtype = self.linear_relpos.weight.dtype

        residue_index = residue_index.long()
        d = residue_index.unsqueeze(-1) - residue_index.unsqueeze(-2)
        d = torch.clamp(d, -self.vbins, self.vbins) + self.vbins
        d_onehot = nn.functional.onehot(d, num_classes=2*self.vbins+1).to(dtype=dtype)
        out = self.linear_relpos(d_onehot)

        return out

    def forward(self, batch: dict) -> tuple[Tensor, Tensor]:
        """
        Implements the forward pass for Algorithm 3.

        Args:
            batch (dict): Feature dictionary with the following entries:
                * msa_feat: Initial MSA feature of shape (*, N_seq, N_res, msa_feat_dim).
                * target_feat: Target feature of shape (*, N_res, tf_dim).
                * residue_index: Residue index of shape (*, N_res)

        Returns:
            tuple: Tuple consisting of the MSA representation m and the pair representation z.
        """

        msa_feat = batch['msa_feat']
        target_feat = batch['target_feat']
        residue_index = batch['residue_index']

        a = self.linear_tf_z_i(target_feat)
        b = self.linear_tf_z_j(target_feat)
        z = a.unsqueeze(-2) + b.unsqueeze(-3)

        z = z + self.relpos(residue_index)
        target_feat = target_feat.unsqueeze(-3)
        m = self.linear_msa_m(msa_feat) + self.linear_tf_m(target_feat)

        return m, z