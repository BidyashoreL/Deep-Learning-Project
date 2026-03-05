# ==========================================
# FINAL INTEGRATED URBAN FLOOD IMPACT MAP
# WITH RIVERS + BALANCED CLASSIFICATION
# ==========================================

import numpy as np
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx

from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.plot import show
from rasterio.features import shapes
from shapely.geometry import box
from matplotlib.colors import ListedColormap

ctx.set_cache_dir("tile_cache")

# ==============================
# ==============================
# FILE PATHS
# ==============================

# ==============================
# FILE PATHS
# ==============================

# ==============================
# FILE PATHS
# ==============================

flood_path = "outputs/vj_flood_risk_classes.tif"

roads_path = "gis/gis_osm_roads_free_1.shp"

water_path = "gis/gis_osm_waterways_free_1.shp"

# ==============================
# REPROJECT FLOOD → EPSG:3857
# ==============================
def reproject_raster(path):

    with rasterio.open(path) as src:

        transform, width, height = calculate_default_transform(
            src.crs, "EPSG:3857",
            src.width, src.height,
            *src.bounds
        )

        data = np.empty((height, width), dtype=np.float32)

        reproject(
            source=rasterio.band(src, 1),
            destination=data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs="EPSG:3857",
            resampling=Resampling.bilinear
        )

        bounds_3857 = rasterio.transform.array_bounds(height, width, transform)

    return data, transform, bounds_3857


flood, flood_transform, flood_bounds = reproject_raster(flood_path)

# ==============================
# NORMALIZE SAFELY
# ==============================
flood = np.nan_to_num(flood)

if flood.max() != flood.min():
    flood = (flood - flood.min()) / (flood.max() - flood.min())

# ==============================
# BALANCED THRESHOLDS
# ==============================
p33 = np.percentile(flood, 33)
p66 = np.percentile(flood, 66)

severity = np.zeros_like(flood)

severity[(flood > p33) & (flood <= p66)] = 2
severity[flood > p66] = 3
severity[severity == 0] = 1

# ==============================
# CLIP BOUNDARY
# ==============================
bbox = box(*flood_bounds)
clip_gdf = gpd.GeoDataFrame(geometry=[bbox], crs="EPSG:3857")

# ==============================
# LOAD & CLIP ROADS
# ==============================
roads = gpd.read_file(roads_path).to_crs(3857)
roads = gpd.clip(roads, clip_gdf)

major = roads[roads["fclass"].isin(["primary", "trunk", "motorway"])]
secondary = roads[~roads["fclass"].isin(["primary", "trunk", "motorway"])]

# ==============================
# LOAD & CLIP WATERWAYS 🌊
# ==============================
water = gpd.read_file(water_path).to_crs(3857)
water = gpd.clip(water, clip_gdf)

# ==============================
# IMPACT ZONE
# ==============================
mask = flood > p66

results = (
    {"properties": {"val": v}, "geometry": s}
    for s, v in shapes(mask.astype(np.uint8), transform=flood_transform)
    if v == 1
)

impact = gpd.GeoDataFrame.from_features(list(results), crs="EPSG:3857")

if not impact.empty:
    impact = impact.buffer(120)

# ==============================
# PLOT
# ==============================
fig, ax = plt.subplots(figsize=(12, 12))

ax.set_xlim(flood_bounds[0], flood_bounds[2])
ax.set_ylim(flood_bounds[1], flood_bounds[3])

# BASEMAP
ctx.add_basemap(
    ax,
    source=ctx.providers.CartoDB.Positron,
    zoom=13,
    alpha=0.35
)

# FLOOD SEVERITY
cmap = ListedColormap(["lightgrey", "orange", "red"])

show(
    severity,
    transform=flood_transform,
    cmap=cmap,
    vmin=1,
    vmax=3,
    alpha=0.75,
    ax=ax
)

# IMPACT ZONE
if not impact.empty:
    impact.plot(
        ax=ax,
        color="darkred",
        alpha=0.25,
        edgecolor="none"
    )

# 🌊 WATERWAYS
water.plot(
    ax=ax,
    color="dodgerblue",
    linewidth=1.2,
    alpha=0.9,
    label="Rivers & Canals"
)

# ROADS
secondary.plot(ax=ax, linewidth=0.6, color="black", alpha=0.5)
major.plot(ax=ax, linewidth=2.2, color="black", label="Major Roads")

# TITLE
ax.set_title(
    "Urban Flood Susceptibility & Impact Zones – Vijayawada",
    fontsize=16,
    weight="bold"
)

ax.axis("off")
ax.legend()

# SAVE
output_file = "final_flood_impact_map.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")

print(f"✅ FINAL MAP SAVED → {output_file}")

# SHOW
plt.show()