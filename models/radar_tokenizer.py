import torch
import torch.nn as nn

class RadarTokenizer(nn.Module):
    """
    Patch tokenizer for radar (ViT-style).
    Converts radar map [B,1,H,W] → [B, N_patches, embed_dim]
    """
    def __init__(self, in_channels=1, embed_dim=768, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)             # [B, embed_dim, H/P, W/P]
        x = x.flatten(2)             # [B, embed_dim, N]
        x = x.transpose(1, 2)        # [B, N, embed_dim]
        return x
