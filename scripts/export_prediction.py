import os
import numpy as np
import rasterio
from tensorflow.keras.models import load_model

# 1. SETUP ABSOLUTE PATHS
BASE_DIR = "/Users/bidyashorelourembam/Deep learning_ Project"
model_path = os.path.join(BASE_DIR, "vj_flood_cnn.h5")
data_path = os.path.join(BASE_DIR, "vj_cnn_input.npy")
dem_path = os.path.join(BASE_DIR, "vijayawada_dem.tif")
output_path = os.path.join(BASE_DIR, "vj_final_prediction.tif")

# 2. LOAD MODEL AND DATA
print(f"🔄 Loading model from: {model_path}")
model = load_model(model_path)

# Load and crop to match the 668x692 training size
X = np.load(data_path)[:668, :692, :]
X_input = np.expand_dims(X, axis=0)

# 3. GENERATE PREDICTION
print("🔮 Predicting Flood Susceptibility...")
prediction = model.predict(X_input)[0, :, :, 0]

# 4. EXPORT AS GEOTIFF
# We use the metadata from your original DEM so it stays georeferenced
with rasterio.open(dem_path) as src:
    meta = src.meta.copy()
    meta.update({
        "driver": "GTiff",
        "height": 668,
        "width": 692,
        "count": 1,
        "dtype": 'float32'
    })

    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(prediction.astype('float32'), 1)

print(f"✅ FINAL SUCCESS: Your map is ready at: {output_path}")