import rasterio
import numpy as np
import geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import shape
import matplotlib.pyplot as plt

flood_risk = "vj_flood_risk_classes.tif"

with rasterio.open(flood_risk) as src:
    data = src.read(1)
    transform = src.transform
    crs = src.crs

# Extract HIGH RISK (value = 3)
mask = data == 3

# Convert to polygons
results = (
    {'properties': {'risk': v}, 'geometry': s}
    for i, (s, v) in enumerate(
        shapes(mask.astype(np.uint8), transform=transform)
    ) if v == 1
)

gdf = gpd.GeoDataFrame.from_features(list(results), crs=crs)

# Buffer → flood spread distance (in meters)
impact_zone = gdf.to_crs(3857).buffer(120).to_crs(crs)

# Plot
fig, ax = plt.subplots(figsize=(10,10))
impact_zone.plot(ax=ax, color='red', alpha=0.4, edgecolor='darkred')

plt.title("Urban Flood Impact Zone")
plt.axis("off")
plt.show()