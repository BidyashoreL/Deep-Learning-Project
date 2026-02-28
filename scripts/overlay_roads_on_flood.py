import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import contextily as ctx
import numpy as np
import os
from matplotlib.colors import LightSource

BASE_DIR = "/Users/bidyashorelourembam/Deep learning_ Project"

flood_path = os.path.join(BASE_DIR, "vj_final_prediction.tif")
roads_path = os.path.join(BASE_DIR, "gis_osm_roads_free_1.shp")
dem_path = os.path.join(BASE_DIR, "vijayawada_dem.tif")

# ---------------- LOAD FLOOD ----------------
with rasterio.open(flood_path) as src:
    flood = src.read(1)
    bounds = src.bounds
    raster_crs = src.crs

# ---------------- LOAD DEM FOR TERRAIN ----------------
with rasterio.open(dem_path) as dem_src:
    dem = dem_src.read(1)

# Create hillshade
ls = LightSource(azdeg=315, altdeg=45)
hillshade = ls.hillshade(dem, vert_exag=1, dx=1, dy=1)

# ---------------- LOAD & FILTER ROADS ----------------
roads = gpd.read_file(roads_path)

major = roads[roads["fclass"].isin(["primary", "secondary", "tertiary"])]

major = major.to_crs(raster_crs)
major = major.cx[bounds.left:bounds.right, bounds.bottom:bounds.top]

# ---------------- PLOT ----------------
fig, ax = plt.subplots(figsize=(12, 12))

# Terrain (hillshade)
ax.imshow(
    hillshade,
    cmap="gray",
    extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
    alpha=0.35
)

# Basemap
ctx.add_basemap(ax, crs=raster_crs, source=ctx.providers.CartoDB.Positron)

# Flood layer
ax.imshow(
    flood,
    cmap="RdYlBu_r",
    alpha=0.5,
    extent=[bounds.left, bounds.right, bounds.bottom, bounds.top]
)

# Styled roads
major[major["fclass"] == "primary"].plot(ax=ax, linewidth=2, color="red")
major[major["fclass"] == "secondary"].plot(ax=ax, linewidth=1.5, color="orange")
major[major["fclass"] == "tertiary"].plot(ax=ax, linewidth=1, color="yellow")

ax.set_title("Urban Flood Risk with Terrain & Transport Network", fontsize=16)
ax.axis("off")

plt.show()