# ============================================================
#   U2Fusion for Multi-Focus Image Fusion (Version A)
#   – Fully working & stable training version
#   – Without skimage, pure PyTorch SSIM
#   – Fix all structural bugs in your program
# ============================================================

import os
import cv2
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Random Seed
# ============================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
seed_everything()


# ============================================================
# Dataset
# ============================================================
class PairDataset(Dataset):
    def __init__(self, pairs, size=256, augment=True):
        self.pairs = pairs
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def load_img(self, p):
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))
        img = img.astype(np.float32) / 255.0
        return img

    def augment_sync(self, a, b):
        if random.random() > 0.5:
            a = cv2.flip(a, 1)
            b = cv2.flip(b, 1)
        return a, b

    def __getitem__(self, idx):
        pa, pb = self.pairs[idx]

        a = self.load_img(pa)
        b = self.load_img(pb)

        if self.augment:
            a, b = self.augment_sync(a, b)

        a = torch.from_numpy(a).permute(2, 0, 1)
        b = torch.from_numpy(b).permute(2, 0, 1)
        return a, b


# ============================================================
# Simple U2Fusion-like CNN
# ============================================================
class U2Fusion(nn.Module):
    def __init__(self, ch=3):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(cout, cout, 3, padding=1),
                nn.ReLU(True),
            )

        self.enc1 = block(ch, 32)
        self.down1 = nn.Conv2d(32, 64, 3, stride=2, padding=1)

        self.enc2 = block(64, 64)
        self.down2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)

        self.bottleneck = block(128, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.dec2 = nn.Conv2d(64, 64, 3, padding=1)

        self.up1 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.dec1 = nn.Conv2d(32, 32, 3, padding=1)

        self.out = nn.Conv2d(32, ch, 3, padding=1)
        self.act = nn.Sigmoid()

        self.fuse1 = nn.Conv2d(32 * 2, 32, 1)
        self.fuse2 = nn.Conv2d(64 * 2, 64, 1)
        self.fuse3 = nn.Conv2d(128 * 2, 128, 1)

    def extract(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.down1(x1))
        x3 = self.bottleneck(self.down2(x2))
        return x1, x2, x3

    def forward(self, a, b):
        a1, a2, a3 = self.extract(a)
        b1, b2, b3 = self.extract(b)

        f1 = self.fuse1(torch.cat([a1, b1], dim=1))
        f2 = self.fuse2(torch.cat([a2, b2], dim=1))
        f3 = self.fuse3(torch.cat([a3, b3], dim=1))

        x = self.up2(f3)
        x = F.relu(self.dec2(x + f2))

        x = self.up1(x)
        x = F.relu(self.dec1(x + f1))

        x = self.act(self.out(x))
        return x


# ============================================================
# SSIM (differentiable)
# ============================================================
def ssim(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)

    sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x ** 2
    sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y

    ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    ssim_map = ssim_n / (ssim_d + 1e-8)
    return torch.clamp((1 - ssim_map.mean()) / 2, 0, 1)


# ============================================================
# Gradient Loss
# ============================================================
def gradient(x):
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32, device=x.device)
    sobel_y = sobel_x.t()

    sobel_x = sobel_x.expand(x.size(1), 1, 3, 3)
    sobel_y = sobel_y.expand(x.size(1), 1, 3, 3)

    gx = F.conv2d(x, sobel_x, padding=1, groups=x.size(1))
    gy = F.conv2d(x, sobel_y, padding=1, groups=x.size(1))

    g = torch.sqrt(gx * gx + gy * gy + 1e-6)
    return g


class FusionLoss(nn.Module):
    def __init__(self, w_grad=10.0):
        super().__init__()
        self.w_grad = w_grad

    def forward(self, fused, a, b):
        loss_ssim = (ssim(fused, a) + ssim(fused, b)) / 2

        ga, gb = gradient(a), gradient(b)
        gt = torch.max(ga, gb)
        gf = gradient(fused)

        loss_grad = F.l1_loss(gf, gt)

        return loss_ssim + self.w_grad * loss_grad


# ============================================================
# Train
# ============================================================
def train(model, loader, device, save="modelA.pth",
          epochs=60):
    model = model.to(device)
    crit = FusionLoss().to(device)
    optimz = torch.optim.Adam(model.parameters(), lr=2e-4)

    best = 999

    for ep in range(1, epochs + 1):
        model.train()
        running = 0

        for a, b in tqdm(loader, desc=f"Epoch {ep}/{epochs}"):
            a, b = a.to(device), b.to(device)

            out = model(a, b)
            loss = crit(out, a, b)

            optimz.zero_grad()
            loss.backward()
            optimz.step()

            running += loss.item()

        avg = running / len(loader)
        print(f"Epoch {ep} Loss={avg:.6f}")

        if avg < best:
            best = avg
            torch.save(model.state_dict(), save)
            print("   Saved best model.")

    return model


# ============================================================
# Inference for 3 images
# ============================================================
def infer_triple(model, paths, size=256, device="cpu"):
    def load(p):
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size))
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    a = load(paths[0])
    b = load(paths[1])
    c = load(paths[2])

    with torch.no_grad():
        f1 = model(a, b)
        f2 = model(f1, c)

    out = f2.squeeze().permute(1, 2, 0).cpu().numpy()
    out = (out * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


# ============================================================
# Main
# ============================================================
file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'res')
triple_dir = os.path.join(file_path, "Triple Series")

if __name__ == "__main__":
    # 构建 pair 列表
    pairs = []
    for i in range(1, 21):
        a = f"{file_path}/lytro-{i:02d}-a.jpg"
        b = f"{file_path}/lytro-{i:02d}-b.jpg"
        if os.path.exists(a) and os.path.exists(b):
            pairs.append([a, b])

    dataset = PairDataset(pairs)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using", device)

    model = U2Fusion()
    model = train(model, loader, device, save="modelA.pth")

    # 推理
    test_paths = [
        f"{triple_dir}/lytro-01-a.jpg",
        f"{triple_dir}/lytro-01-b.jpg",
        f"{triple_dir}/lytro-01-c.jpg",
    ]
    out = infer_triple(model, test_paths, device=device)
    cv2.imwrite("fused_A.jpg", out)
    print("Saved fused_A.jpg")
