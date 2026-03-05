import torch
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from src.models.unet import UNet

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Running on:", DEVICE)

paths = {
    "rain": "data/vj_rainfall_dynamic.tif",
    "dem": "data/vijayawada_dem.tif",
    "slope": "data/slope.tif",
    "drain": "data/vj_drainage_distance.tif",
    "lulc": "data/vijayawada_lulc.tif",
}

with rasterio.open(paths["dem"]) as ref:
    ref_meta = ref.meta.copy()
    ref_transform = ref.transform
    ref_crs = ref.crs
    H, W = ref.height, ref.width

stack = []

for key in paths:
    with rasterio.open(paths[key]) as src:

        data = np.empty((H, W), dtype=np.float32)

        reproject(
            source=rasterio.band(src, 1),
            destination=data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear,
        )

        # 🔥 SAME normalization
        if data.max() != data.min():
            data = (data - data.min()) / (data.max() - data.min())

        stack.append(data)

X = np.stack(stack)

x = torch.tensor(X).unsqueeze(0).to(DEVICE)

_, _, h, w = x.shape
pad_h = (16 - h % 16) % 16
pad_w = (16 - w % 16) % 16
x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))

model = UNet(in_channels=5, num_classes=3).to(DEVICE)
model.load_state_dict(torch.load("model/best_unet.pth", map_location=DEVICE))
model.eval()

with torch.no_grad():
    output = model(x)
    prediction = torch.argmax(output, dim=1)

prediction = prediction[:, :h, :w]
pred_np = prediction.squeeze().cpu().numpy().astype(np.uint8)

print("Class distribution:", np.unique(pred_np, return_counts=True))

ref_meta.update(dtype="uint8", count=1, nodata=0)

with rasterio.open("outputs/unet_flood_prediction.tif", "w", **ref_meta) as dst:
    dst.write(pred_np, 1)

print("✅ Prediction saved")