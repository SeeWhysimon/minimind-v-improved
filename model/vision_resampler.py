import torch
import torch.nn as nn

from typing import Optional


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
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
            batch_first=True,
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
        query: torch.Tensor,
        context: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query:
                形状为 [batch_size, query_length, embed_dim]

            context:
                形状为 [batch_size, context_length, embed_dim]

            key_padding_mask:
                形状为 [batch_size, context_length]
                True 表示该位置不参与注意力计算

        Returns:
            与 q 形状相同的张量。
        """

        q_normed = self.q_norm(query)
        context_normed = self.kv_norm(context)

        attention_output, _ = self.cross_attention(
            query=q_normed,
            key=context_normed,
            value=context_normed,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        hidden = query + self.attention_dropout(attention_output)
        output = hidden + self.ffn_dropout(self.ffn(self.ffn_norm(hidden)))

        return output
    

class VisionTokenResampler(nn.Module):
    def init_weight(self, ):
        pass
    
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        num_queries: int = 64, 
        num_heads: int = 8, 
        depth: int = 1, 
        mlp_ratio: float = 4.0, 
        dropout: float = 0.2, 
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(in_features=input_dim, out_features=output_dim) if input_dim != output_dim else nn.Identity()
        self.attention_blocks = nn.ModuleList([
            CrossAttentionBlock(output_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.learned_queries = nn.Parameter(torch.randn(1, num_queries, output_dim) * 0.02)
        self.output_norm = nn.LayerNorm(output_dim)
        
        self.init_weight()
        
    def forward(
        self, 
        vision_tokens: torch.Tensor, 
        resampler_key_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            vision_tokens:
                形状为 [batch_size, context_length, input_dim]

            vision_mask:
                形状为 [batch_size, context_length]
                True 表示有效视觉 token
                False 表示 padding token

        Returns:
            qeury:
                形状为 [batch_size, num_queries, output_dim]
        """
        if vision_tokens.ndim != 3:
            raise ValueError("错误的 vision_token 维度")
        
        context = self.input_proj(vision_tokens)
        
        batch_size, context_len, _ = vision_tokens.shape
        queries = self.learned_queries.expand(batch_size, -1, -1)
        
        for block in self.attention_blocks:
            queries = block(queries, context, resampler_key_padding_mask)
        
        queries = self.output_norm(queries)
        return queries  