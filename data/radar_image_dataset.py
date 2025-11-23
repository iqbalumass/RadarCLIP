import os
import csv
from torch.utils.data import Dataset
from PIL import Image, ImageFile

# FIX for truncated DeepSense6G RGB files
ImageFile.LOAD_TRUNCATED_IMAGES = True

from PIL import Image
from .radar_loader import load_radar_mat

class RadarImageDataset(Dataset):
    def __init__(self, csv_path, base_path="", img_transform=None,
                 img_col="unit1_rgb1", radar_col="unit1_radar1", radar_size=224):
        self.base_path = base_path
        self.img_transform = img_transform
        self.radar_size = radar_size
        self.img_col = img_col
        self.radar_col = radar_col

        self.samples = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = os.path.join(base_path, row[img_col])
                radar_path = os.path.join(base_path, row[radar_col])
                if os.path.exists(img_path) and os.path.exists(radar_path):
                    self.samples.append((img_path, radar_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, radar_path = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        if self.img_transform:
            img = self.img_transform(img)

        radar = load_radar_mat(radar_path, size=self.radar_size)

        return img, radar
