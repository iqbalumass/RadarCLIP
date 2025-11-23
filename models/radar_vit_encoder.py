import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .radar_tokenizer import RadarTokenizer

class RadarViTEncoder(nn.Module):
    """
    ViT-style transformer encoder for radar embeddings.
    Outputs a CLIP-compatible embedding.
    """
    def __init__(
        self,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        patch_size=16,
        in_channels=1,
        out_clip_dim=512
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.tokenizer = RadarTokenizer(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = None  # allocated dynamically

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.ln_post = nn.LayerNorm(embed_dim)
        #self.proj = nn.Linear(embed_dim, out_clip_dim)
        self.proj = nn.Sequential(
           nn.Linear(embed_dim, embed_dim),
           nn.GELU(),
           nn.Linear(embed_dim, out_clip_dim),
        )

    def _build_pos_embed(self, N, device):
        if self.pos_embed is None or self.pos_embed.size(1) != N + 1:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, N + 1, self.embed_dim, device=device)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.size(0)

        tokens = self.tokenizer(x)       # [B, N, D]
        N = tokens.size(1)

        self._build_pos_embed(N, x.device)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, tokens], dim=1) + self.pos_embed

        x = self.transformer(x)
        x = self.ln_post(x[:, 0])        # CLS token
        x = self.proj(x)
        return x
