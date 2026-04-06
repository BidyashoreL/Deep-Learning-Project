import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling
import os

paths = {
    "rain": "data/vj_rainfall_dynamic.tif",
    "dem": "data/vijayawada_dem.tif",
    "slope": "data/slope.tif",
    "drain": "data/vj_drainage_distance.tif",
    "lulc": "data/vijayawada_lulc.tif",
}

# --------------------------------------------------
# USE DEM AS REFERENCE GRID
# --------------------------------------------------
with rasterio.open(paths["dem"]) as ref:
    ref_meta = ref.meta.copy()
    ref_transform = ref.transform
    ref_crs = ref.crs
    H, W = ref.height, ref.width

stack = []

# --------------------------------------------------
# ALIGN ALL RASTERS
# --------------------------------------------------
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

        # Normalize
        if data.max() != data.min():
            data = (data - data.min()) / (data.max() - data.min())

        stack.append(data)

# Stack (5, H, W)
stack = np.stack(stack)

# --------------------------------------------------
# SAVE
# --------------------------------------------------
ref_meta.update({
    "count": 5,
    "dtype": "float32"
})

os.makedirs("outputs", exist_ok=True)

with rasterio.open("outputs/input_stack.tif", "w", **ref_meta) as dst:
    dst.write(stack)

print("✅ input_stack.tif created successfully")