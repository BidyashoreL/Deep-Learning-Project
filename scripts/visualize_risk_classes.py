import rasterio
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import ListedColormap

BASE_DIR = "/Users/bidyashorelourembam/Deep learning_ Project"
risk_path = os.path.join(BASE_DIR, "vj_flood_risk_classes.tif")

with rasterio.open(risk_path) as src:
    risk = src.read(1)

cmap = ListedColormap(["green", "yellow", "red"])

plt.figure(figsize=(8,8))
plt.imshow(risk, cmap=cmap)
plt.title("Urban Flood Risk Zones")
plt.axis("off")

labels = ["Low", "Moderate", "High"]
colors = ["green", "yellow", "red"]

patches = [plt.plot([],[], marker="s", ms=10, ls="", mec=None, color=c)[0] for c in colors]
plt.legend(patches, labels, loc="lower right")

plt.show()