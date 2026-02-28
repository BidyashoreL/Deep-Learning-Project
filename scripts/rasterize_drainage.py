import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
import os

BASE_DIR = "/Users/bidyashorelourembam/Deep learning_ Project"
dem_path = os.path.join(BASE_DIR, "vijayawada_dem.tif")
# Point this to the file you just downloaded
shapefile_path = os.path.join(BASE_DIR, "gis_osm_waterways_free_1.shp")

print("🌊 Loading local drainage shapefile...")
waterways = gpd.read_file(shapefile_path)

# 1. Load DEM metadata for alignment
with rasterio.open(dem_path) as src:
    meta = src.meta
    master_shape = src.shape
    bounds = src.bounds

# 2. Clip waterways to your DEM extent so it's fast
print("✂️ Clipping waterways to Vijayawada area...")
waterways = waterways.cx[bounds.left:bounds.right, bounds.bottom:bounds.top]

# 3. Rasterize: 1 where there is a canal/river, 0 otherwise
print("🎨 Converting vectors to raster...")
shapes = [(geom, 1) for geom in waterways.geometry]
drain_mask = rasterize(shapes, out_shape=master_shape, transform=meta['transform'])

# 4. Calculate Distance (Proximity)
print("📏 Calculating distance to nearest drainage...")
# This gives the distance in pixels; multiplying by 30 gives approx meters
dist_map = distance_transform_edt(drain_mask == 0)

# 5. Save the final layer
meta.update(dtype='float32', count=1)
output_path = os.path.join(BASE_DIR, "vj_drainage_distance.tif")
with rasterio.open(output_path, 'w', **meta) as dst:
    dst.write(dist_map.astype('float32'), 1)

print(f"✅ SUCCESS: {output_path} created locally!")