import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

path = "outputs/unet_flood_prediction.tif"

with rasterio.open(path) as src:
    data = src.read(1)

cmap = ListedColormap(["lightgrey", "orange", "red"])

plt.figure(figsize=(10, 8))
plt.imshow(data, cmap=cmap, vmin=1, vmax=3)
plt.title("U-Net Urban Flood Susceptibility Map")
plt.axis("off")
plt.show()