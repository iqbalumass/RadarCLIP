import numpy as np
import torch
from scipy.io import loadmat
import torch.nn.functional as F

def load_radar_mat(path, size=224):
    data = loadmat(path, simplify_cells=True)
    key = [k for k in data.keys() if not k.startswith("__")][0]
    arr = data[key]

    # Convert complex → magnitude
    if np.iscomplexobj(arr):
        arr = np.abs(arr)

    ra = np.array(arr, dtype=np.float32)

    # -----------------------------------------
    # 1) Handle ANY radar shape:
    #    (H, W)
    #    (C, H, W) → e.g., DeepSense6G radar cube
    #    (H, W, C) → less common but possible
    # -----------------------------------------
    if ra.ndim == 3:
        # Option A: mean across channels
        ra = ra.mean(axis=0)

    elif ra.ndim > 3:
        # Flatten all leading dims except the last two
        ra = ra.reshape(ra.shape[-2], ra.shape[-1])

    # -----------------------------------------
    # 2) Standardize radar (important for CLIP alignment)
    # -----------------------------------------
    ra = (ra - ra.mean()) / (ra.std() + 1e-8)

    # Convert to torch [1, 1, H, W]
    radar = torch.from_numpy(ra).unsqueeze(0).unsqueeze(0)

    # -----------------------------------------
    # 3) Resize safely now that it's 2D
    # -----------------------------------------
    radar = F.interpolate(
        radar, size=(size, size), mode="bilinear", align_corners=False
    )

    return radar.squeeze(0)  # [1, size, size]
