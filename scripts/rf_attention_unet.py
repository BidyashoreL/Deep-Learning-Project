"""
RF-AttentionUNet: Hybrid U-Net + Random Forest Attention
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import rasterio
import os


# ============================================================
# 1. BASIC BLOCKS
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class RFAttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int, rf_dim=5):
        super().__init__()

        self.W_g = nn.Conv2d(F_g, F_int, 1)
        self.W_x = nn.Conv2d(F_l, F_int, 1)

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1),
            nn.Sigmoid()
        )

        self.rf_gate = nn.Sequential(
            nn.Linear(rf_dim, F_int),
            nn.ReLU(),
            nn.Linear(F_int, 1),
            nn.Sigmoid()
        )

    def forward(self, g, x, rf_imp):

        g1 = self.W_g(g)
        x1 = self.W_x(x)

        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear')

        psi = self.psi(F.relu(g1 + x1))

        G = self.rf_gate(rf_imp).unsqueeze(-1).unsqueeze(-1)

        att = psi * G
        return x * att


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, rf_dim=5):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.gate = RFAttentionGate(out_ch, skip_ch, out_ch // 2, rf_dim)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip, rf_imp):
        x = self.up(x)
        skip = self.gate(x, skip, rf_imp)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ============================================================
# 2. RF BRANCH
# ============================================================

class RFBranch:
    def __init__(self):
        self.rf = RandomForestClassifier(n_estimators=100)
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, X, y):
        X = self.scaler.fit_transform(X)
        self.rf.fit(X, y)
        self.fitted = True

    def get_importance(self, batch):
        imp = torch.tensor(self.rf.feature_importances_, dtype=torch.float32)
        return imp.unsqueeze(0).repeat(batch, 1)


# ============================================================
# 3. MODEL
# ============================================================

class RFAttentionUNet(nn.Module):
    def __init__(self, in_ch=5, num_classes=3):
        super().__init__()

        self.enc1 = ConvBlock(in_ch, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.enc4 = ConvBlock(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(256, 512)

        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 32, 32)

        self.head = nn.Conv2d(32, num_classes, 1)

    def forward(self, x, rf_imp):

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(b, e4, rf_imp)
        d3 = self.dec3(d4, e3, rf_imp)
        d2 = self.dec2(d3, e2, rf_imp)
        d1 = self.dec1(d2, e1, rf_imp)

        return self.head(d1)


# ============================================================
# 4. TRAINER
# ============================================================

class Trainer:
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        self.model = RFAttentionUNet().to(self.device)
        self.rf = RFBranch()

        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self, image, label, patches, labels):

        # FIX SIZE
        H = min(image.shape[1], label.shape[0])
        W = min(image.shape[2], label.shape[1])

        image = image[:, :H, :W]
        label = label[:H, :W]

        # RF TRAIN
        X = image.reshape(image.shape[0], -1).T
        y = label.flatten()

        self.rf.fit(X, y)

        rf_imp = self.rf.get_importance(1).to(self.device)

        # TRAIN
        for epoch in range(10):
            total = 0

            for p, l in zip(patches, labels):
                p = p.unsqueeze(0).to(self.device)
                l = l.unsqueeze(0).to(self.device)

                self.opt.zero_grad()
                out = self.model(p, rf_imp)
                loss = self.loss_fn(out, l)
                loss.backward()
                self.opt.step()

                total += loss.item()

            avg_loss = total / len(patches)
            print(f"Epoch {epoch+1}: {avg_loss:.4f}")

# ============================================================
# 5. MAIN
# ============================================================

if __name__ == "__main__":

    print("Loading data...")

    img_path = "outputs/input_stack.tif"
    lbl_path = "data/label.tif"

    with rasterio.open(img_path) as src:
        image = src.read().astype(np.float32)

    with rasterio.open(lbl_path) as src:
        label = src.read(1).astype(np.int64)

    # normalize
    for c in range(image.shape[0]):
        mn, mx = image[c].min(), image[c].max()
        if mx > mn:
            image[c] = (image[c] - mn) / (mx - mn)

    # patches
    PATCH = 128
    STRIDE = 64

    patches, labels = [], []

    H, W = label.shape

    for y in range(0, H - PATCH, STRIDE):
        for x in range(0, W - PATCH, STRIDE):
            patches.append(torch.tensor(image[:, y:y+PATCH, x:x+PATCH]))
            labels.append(torch.tensor(label[y:y+PATCH, x:x+PATCH]))

    print("Dataset:", len(patches))

    trainer = Trainer()
    trainer.train(image, label, patches, labels)