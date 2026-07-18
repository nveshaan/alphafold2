import torch.nn as nn
from src.attention.mha import MultiHeadAttention
from src.evoformer.dropout import DropoutRowwise
from src.evoformer.msa_stack import MSARowAttentionWithPairBias, MSATransition, OuterProductMean
from src.evoformer.pair_stack import PairStack
from torch import Tensor


class ExtraMsaEmbedder(nn.Module):
    """
    Creates the embeddings of extra_msa_feat for the Extra MSA Stack.
    """
    
    def __init__(self, f_e: int, c_e: int):
        """
        Initializes the ExtraMSAEmbedder.

        Args:
            f_e (int): Initial dimension of the extra_msa_feat.
            c_e (int): Embedding dimension of the extra_msa_feat.
        """
        super().__init__()
        self.linear = nn.Linear(f_e, c_e)

    def forward(self, batch: dict) -> Tensor:
        """
        Passes extra_msa_feat through a linear embedder.

        Args:
            batch (dict): Feature dictionary with the following entries:
                * extra_msa_feat: Extra MSA feature of shape (*, N_extra, N_res, f_e).

        Returns:
            torch.tensor: Output tensor of shape (*, N_extra, N_res, c_e):
        """

        e = batch['extra_msa_feat']
        out = self.linear(e)

        return out


class MSAColumnGlobalAttention(nn.Module):
    """
    Implements Algorithm 19.
    """
    
    def __init__(self, c_m: int, c_z: int, c: int=8, N_head: int=8):
        """
        Initializes MSAColumnGlobalAttention.

        Args:
            c_m (int): Embedding dimension of the MSA representation.
            c_z (int): Embedding dimension of the pair representation.
            c (int, optional): Embedding dimension for MultiHeadAttention. Defaults to 8.
            N_head (int, optional): Number of heads for MultiHeadAttention. Defaults to 8.
        """

        super().__init__()

        self.layer_norm_m = nn.LayerNorm(c_m)
        self.global_attention = MultiHeadAttention(c_m, c, attn_dim=-3, N_head=N_head, gated=True, is_global=True)

    def forward(self, m: Tensor) -> Tensor:
        """
        Implements the forward pass for Algorithm 19.

        Args:
            m (torch.tensor): MSA representation of shape (*, N_seq, N_res, c_m).

        Returns:
            torch.tensor: Output tensor of the same shape as m.
        """

        m = self.layer_norm_m(m)
        out = self.global_attention(m)

        return out


class ExtraMsaBlock(nn.Module):
    """
    Implements one block for Algorithm 18.
    """
    
    def __init__(self, c_e: int, c_z: int):
        """
        Initializes ExtraMSABlock.

        Args:
            c_e (int): Embedding dimension of the extra MSA representation.
            c_z (int): Embedding dimension of the pair representation.
        """
        super().__init__()

        self.dropout_rowwise = DropoutRowwise(p=0.15)
        self.msa_att_row = MSARowAttentionWithPairBias(c_e, c_z, c=8)
        self.msa_att_col = MSAColumnGlobalAttention(c_e, c_z)
        self.msa_transition = MSATransition(c_e)
        self.outer_product_mean = OuterProductMean(c_e, c_z)
        self.core = PairStack(c_z)

    def forward(self, e: Tensor, z:Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward pass for Algorithm 18.

        Args:
            e (torch.tensor): Extra MSA representation of shape (*, N_extra, N_res, c_e).
            z (torch.tensor): Pair representation of shape (*, N_res, N_res, c_z).

        Returns:
            tuple: Tuple consisting of the transformed features e and z.
        """

        e = e + self.dropout_rowwise(self.msa_att_row(e, z))
        e = e + self.msa_att_col(e)
        e = e + self.msa_transition(e)

        z = z + self.outer_product_mean(e)
        z = self.core(z)

        return e, z


class ExtraMsaStack(nn.Module):
    """
    Implements Algorithm 18.
    """

    def __init__(self, c_e: int, c_z: int, num_blocks: int):
        """
        Initializes the ExtraMSAStack.

        Args:
            c_e (int): Embedding dimension of the extra MSA representation.
            c_z (int): Embedding dimension of the pair representation.
            num_blocks (int): Number of blocks in the ExtraMSAStack.
        """
        super().__init__()
        self.blocks = nn.ModuleList([ExtraMsaBlock(c_e, c_z) for _ in range(num_blocks)])

    def forward(self, e: Tensor, z: Tensor) -> Tensor:
        """
        Implements the forward pass for Algorithm 18.

        Args:
            e (torch.tensor): Extra MSA representation of shape (*, N_extra, N_res, c_e).
            z (torch.tensor): Pair representation of shape (*, N_res, N_res, c_z).

        Returns:
            torch.tensor: Output tensor of the same shape as z.
        """

        for msablock in self.blocks:
            e, z = msablock(e, z)

        return z