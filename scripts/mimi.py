import ee
import geemap
import os

# Initialize with your project ID
project_id = 'urban-flood-emergency-response' 
ee.Initialize(project=project_id)

# 1. Define Area (Same as before for consistency)
vj_point = ee.Geometry.Point([80.648, 16.506])
region = vj_point.buffer(10000).bounds()

# 2. Fetch Sentinel-2 Surface Reflectance (LULC Source)
# We filter for the clearest (least cloudy) image from 2024
s2_collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
    .filterBounds(region) \
    .filterDate('2024-01-01', '2024-12-31') \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5)) \
    .median()

# 3. Calculate NDBI (Built-Up Index) 
# Formula: (SWIR - NIR) / (SWIR + NIR)
# High NDBI = Concrete/Buildings
ndbi = s2_collection.normalizedDifference(['B11', 'B8']).rename('NDBI').clip(region)

# 4. Export the LULC Layer
out_file = os.path.join(os.getcwd(), 'vijayawada_lulc.tif')
geemap.ee_export_image(ndbi, filename=out_file, scale=30, region=region)

print(f"🚀 18.5% Importance Layer Saved: {out_file}")