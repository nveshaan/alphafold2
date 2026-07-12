import math
import torch
import torch.nn as nn
from typing import Tuple
from torch import Tensor

class MultiHeadAttention(nn.Module):
    """
    A MultiHeadAttention module with optional bias and optional gating.
    """

    def __init__(self, c_in: int, c, N_head: int, attn_dim: int, gated: bool=False, is_global: bool=False, use_bias_for_embeddings: bool=False):
        """
        MultiHeadAttention theoretically consists of N_head separate linear
        layers for the query, key and value embeddings. However, the embeddings
        can be computed jointly and split afterwards, so we only need one query,
        key and value layer with larger c_out.

        Args:
            c_in (int): Input dimension for the embeddings.
            c (int): Embedding dimension for each individual head.
            N_head (int): Number of heads.
            attn_dim (int): The dimension in the input tensor along which
                the attention mechanism is performed.
            gated (bool, optional): If True, an additional sigmoid-activated
                linear layer will be multiplicated against the weighted
                value vectors before feeding them through the output layer.
                Defaults to False.
            is_global (bool, optional): If True, global calculation will be performed.
                For global calculation, key and value embeddings will only use one head,
                and the q query vectors will be averaged to one query vector.
                Defaults to False.
            use_bias_for_embeddings (bool, optional): If True, query,
                key, and value embeddings will use bias, otherwise not.
                Defaults to False.
        """
        super().__init__()

        self.c_in = c_in
        self.c = c
        self.N_head = N_head
        self.gated = gated
        self.attn_dim = attn_dim
        self.is_global = is_global

        self.linear_q = nn.Linear(c_in, c*N_head, bias=use_bias_for_embeddings)

        c_kv = c if is_global else c*N_head
        self.linear_k = nn.Linear(c_in, c_kv, bias=use_bias_for_embeddings)
        self.linear_v = nn.Linear(c_in, c_kv, bias=use_bias_for_embeddings)

        self.linear_o = nn.Linear(c*N_head, c_in)

        if gated:
            self.linear_g = nn.Linear(c_in, c*N_head)

    def prepare_qkv(self, q: Tensor, k: Tensor, v: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Splits the embeddings into individual heads and transforms the input
        shapes of form (*, q/k/v, *, N_head*c) into the shape
        (*, N_head, q/k/v, *, c). The position of the q/k/v dimension
        in the original tensors is given by attn_dim.

        Args:
            q (Tensor): Query embedding of shape (*, q, *, N_head*c).
            k (Tensor): Key embedding of shape (*, k, *, N_head*c).
            v (Tensor): Value embedding of shape (*, v, *, N_head*c).

        Returns:
            tuple: The rearranged embeddings q, k, and v of
                shape (*, N_head, q/k/v, c) respectively.
        """

        # Transposing to [*, q/k/v, N_head*c]
        q = q.movedim(self.attn_dim, -2)
        k = k.movedim(self.attn_dim, -2)
        v = v.movedim(self.attn_dim, -2)

        # Unwrapping to [*, q/k/v, N_head, c]
        q_shape = q.shape[:-1] + (self.N_head, -1)
        k_shape = k.shape[:-1] + (self.N_head, -1)
        v_shape = v.shape[:-1] + (self.N_head, -1)

        q = q.view(q_shape)
        k = k.view(k_shape)
        v = v.view(v_shape)

        # Transposing to [*, N_head, q/k/v, c]
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        return q, k, v

    def prepare_qkv_global(self, q: Tensor, k: Tensor, v:Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Prepares the query, key and value embeddings with the following
        differences to the non-global version:
            - key and value embeddings use only one head.
            - the query vectors are contracted into one, average query vector.

        Args:
            q (Tensor): Query embeddings of shape (*, q, *, N_head*c).
            k (Tensor): Key embeddings of shape (*, k, *, c).
            v (Tensor): Value embeddings of shape (*, v, *, c).

        Returns:
            tuple: The rearranged embeddings q, k, and v of
                shape (*, N_head, 1, c) for q and shape (*, 1, k, c) for k and v.
        """

        q = q.movedim(self.attn_dim, -2)
        k = k.movedim(self.attn_dim, -2)
        v = v.movedim(self.attn_dim, -2)

        q_shape = q.shape[:-1] + (self.N_head, -1)
        q = q.view(q_shape)

        q = q.transpose(-2, -3)
        k = k.unsqueeze(-3)
        v = v.unsqueeze(-3)

        q = torch.mean(q, dim=-2, keepdim=True)

        return q, k, v

    def forward(self, x: Tensor, bias: Tensor=None, attention_mask: Tensor=None) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (*, q/k/v, *, c_in)
            bias (Tensor, optional): Optional bias tensor of shape
                (*, N_head, q, k) that will be added to the attention weights.
                Defaults to None.
            attention_mask (Tensor, optional): Optional attention mask
                of shape (*, k). If set, the keys with value 0 in the mask will
                not be attended to.

        Returns:
            Tensor: Output tensor of shape (*, q/k/v, *, c_in)
        """
        out = None

        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)

        if self.is_global:
            q, k, v = self.prepare_qkv_global(q, k, v)
        else:
            q, k, v = self.prepare_qkv(q, k, v)

        q = q / math.sqrt(self.c)

        a = torch.einsum('...qc,...kc->...qk', q, k)

        if bias is not None:
            # bias_batch_shape = bias.shape[:-3]
            # bias_bc_shape = bias_batch_shape + (1,)*(a.ndim-len(bias_batch_shape)-3) + bias.shape[-3:]
            # bias = bias.view(bias_bc_shape)
            bias = bias.unsqueeze(1)

            a = a + bias

        if attention_mask is not None:
            attention_mask = attention_mask[..., None, None, :]
            # offset = (attention_mask==0) * -1e8
            # offset = torch.where(attention_mask == 0, -1e8, 0.0)
            offset = attention_mask.masked_fill(attention_mask == 0, -1e8)

            a = a + offset

        a = torch.softmax(a, dim=-1)
        o = torch.einsum('...qk,...kc->...qc', a, v)
        o = o.transpose(-2, -3)
        o = o.flatten(start_dim=-2)
        o = o.movedim(-2, self.attn_dim)

        if self.gated:
            g = torch.sigmoid(self.linear_g(x))
            o = o * g

        out = self.linear_o(o)
        return out