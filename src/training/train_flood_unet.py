import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

from src.dataset.flood_dataset import FloodDataset
from src.models.unet import UNet

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", DEVICE)

raster_paths = [
    "data/vj_rainfall_dynamic.tif",
    "data/vijayawada_dem.tif",
    "data/slope.tif",
    "data/vj_drainage_distance.tif",
    "data/vijayawada_lulc.tif",
]

label_path = "data/label.tif"

dataset = FloodDataset(raster_paths, label_path)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = UNet(in_channels=5, num_classes=3).to(DEVICE)

# 🔥 Balanced weights (NOT extreme)
weights = torch.tensor([1.0, 1.5, 2.0]).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

os.makedirs("model", exist_ok=True)
best_loss = float("inf")

for epoch in range(40):

    model.train()
    total_loss = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "model/best_unet.pth")
        print("⭐ Best model saved")

print("✅ Training complete")