import osmnx as ox
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
import numpy as np

# 1. Load your DEM to get the exact "Geographic Box"
with rasterio.open('vijayawada_dem.tif') as src:
    bounds = src.bounds
    meta = src.meta
    master_shape = src.shape

# 2. UPDATED DOWNLOAD: Use a tuple for the bbox (North, South, East, West)
print("📡 Downloading Vijayawada drainage network (Canals & Rivers)...")
# Note the double parentheses: features_from_bbox((north, south, east, west), ...)
bbox = (bounds.top, bounds.bottom, bounds.right, bounds.left)
graph = ox.features_from_bbox(bbox=bbox, tags={"waterway": ["canal", "river", "stream", "drain"]})

# 3. Convert vectors to a raster "mask"
if graph.empty:
    print("⚠️ No waterways found! Check your DEM coordinates.")
else:
    print(f"✅ Found {len(graph)} waterway segments.")
    shapes = [(geom, 1) for geom in graph.geometry if geom.geom_type in ['LineString', 'MultiLineString']]
    drain_mask = rasterize(shapes, out_shape=master_shape, transform=meta['transform'])

    # 4. Create the Distance Map
    dist_map = distance_transform_edt(drain_mask == 0) 

    # 5. Save as a Georeferenced TIF
    meta.update(dtype='float32', count=1)
    output_path = 'vj_drainage_distance.tif'
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(dist_map.astype('float32'), 1)

    print(f"✅ SUCCESS: {output_path} created and aligned with DEM!")