import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from .radar_vit_encoder import RadarViTEncoder
import numpy as np

class RadarImageCLIP(nn.Module):
    """
    CLIP-style contrastive model:
    Image → pretrained CLIP visual encoder (frozen)
    Radar → RadarViT encoder (trainable)
    """
    def __init__(self, device="cuda"):
        super().__init__()

        self.device = device

        # Load CLIP for images
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="openai", device=device
        )
        self.image_encoder = clip_model.visual
        self.image_preprocess = preprocess

        # Freeze CLIP image encoder
        for p in self.image_encoder.parameters():
            p.requires_grad = False

        self.clip_dim = self.image_encoder.output_dim   # usually 512

        # Radar encoder (trainable)
        self.radar_encoder = RadarViTEncoder(
            embed_dim=768,
            depth=6,
            num_heads=8,
            patch_size=16,
            out_clip_dim=self.clip_dim
        )

        # CLIP logit scale
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))

    def forward(self, img, radar):
        with torch.no_grad():
            img_embed = self.image_encoder(img)  # [B,clip_dim]

        radar_embed = self.radar_encoder(radar)  # [B,clip_dim]

        img_embed = F.normalize(img_embed, dim=-1)
        radar_embed = F.normalize(radar_embed, dim=-1)

        return img_embed, radar_embed, self.logit_scale.exp()
