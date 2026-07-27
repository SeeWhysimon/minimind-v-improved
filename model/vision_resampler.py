import torch
import torch.nn as nn

from typing import Optional

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        

class CrossAttentionSampler(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        batch_first: bool = True,
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) 必须能被 num_heads ({num_heads}) 整除")

        mlp_hidden_dim = int(embed_dim * mlp_ratio)

        self.q_norm = nn.LayerNorm(embed_dim)
        self.kv_norm = nn.LayerNorm(embed_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first,
        )

        self.attention_dropout = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
        )

        self.ffn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            q:
                batch_first=True 时形状为
                [batch_size, query_length, embed_dim]

            k, v:
                batch_first=True 时形状为
                [batch_size, context_length, embed_dim]

            key_padding_mask:
                形状为 [batch_size, context_length]。
                True 表示该位置不参与注意力计算。

            attn_mask:
                注意力掩码，通常形状为
                [query_length, context_length]。

        Returns:
            与 q 形状相同的张量。
        """

        q_normed = self.q_norm(q)
        k_normed = self.kv_norm(k)
        v_normed = self.kv_norm(v)

        attention_output, _ = self.cross_attention(
            query=q_normed,
            key=k_normed,
            value=v_normed,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )

        hidden = q + self.attention_dropout(attention_output)

        output = hidden + self.ffn_dropout(
            self.ffn(self.ffn_norm(hidden))
        )

        return output