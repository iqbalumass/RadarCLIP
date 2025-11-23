import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import open_clip

from models.radar_clip_model import RadarImageCLIP
from data.radar_image_dataset import RadarImageDataset

@torch.no_grad()
def eval_radar_clip(csv_path, base_path, device="cuda", batch_size=32):
    # 1) Build model and load trained radar encoder
    model = RadarImageCLIP(device=device).to(device)
    state = torch.load("radar_vit_clip.pth", map_location=device)
    model.radar_encoder.load_state_dict(state, strict=False)
    model.eval()

    # 2) Dataset + loader (same transforms as training)
    dummy_clip, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="openai", device=device
    )
    ds = RadarImageDataset(
        csv_path=csv_path,
        base_path=base_path,
        img_transform=preprocess,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_img = []
    all_radar = []

    for imgs, radars in loader:
        imgs = imgs.to(device)
        radars = radars.to(device)

        img_embed, radar_embed, _ = model(imgs, radars)
        all_img.append(img_embed)
        all_radar.append(radar_embed)

    all_img = F.normalize(torch.cat(all_img, dim=0), dim=-1)
    all_radar = F.normalize(torch.cat(all_radar, dim=0), dim=-1)

    # 3) Cosine similarity matrix
    sims = all_radar @ all_img.t()  # [N,N]

    # diagonal = correct pairs; off-diagonal = mismatched
    diag = sims.diag()
    off_diag = sims[~torch.eye(sims.size(0), dtype=bool, device=device)]

    print(f"Avg similarity (correct pairs):     {diag.mean().item():.3f}")
    print(f"Avg similarity (mismatched pairs): {off_diag.mean().item():.3f}")

    # simple retrieval accuracy @1
    top1 = sims.argmax(dim=1)
    targets = torch.arange(sims.size(0), device=device)
    acc1 = (top1 == targets).float().mean().item()
    print(f"Top-1 retrieval accuracy: {acc1*100:.2f}%")

if __name__ == "__main__":
    eval_radar_clip(
        csv_path="/work/pi_mshao_umassd_edu/Iqbal/RadarCLIP/scenario36/scenario36.csv",
        base_path="/work/pi_mshao_umassd_edu/Iqbal/RadarCLIP/scenario36/",
    )
