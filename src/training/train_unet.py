import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from src.models.unet import UNet
from src.dataset.flood_dataset import FloodDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = FloodDataset(
    "data/stack.npy",
    "data/mask.npy"
)

loader = DataLoader(dataset, batch_size=1)

model = UNet().to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 20

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    for x, y in loader:

        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss {total_loss:.4f}")

torch.save(model.state_dict(), "model/unet_flood.pth")
print("✅ MODEL TRAINED & SAVED")