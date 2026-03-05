import numpy as np
import rasterio
import matplotlib
# MUST come before pyplot — prevents macOS hang on plt.show()
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from rasterio.plot import show as rio_show
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import box as shapely_box
import contextily as ctx
import os

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

ctx.set_cache_dir("tile_cache")   # cache tiles locally — avoids re-downloading

# ============================================================
# 1. REPROJECT FLOOD RASTER → EPSG:3857
#    (same approach as CNN script — required for ctx basemap)
# ============================================================
def reproject_to_3857(path):
    with rasterio.open(path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, "EPSG:3857",
            src.width, src.height, *src.bounds
        )
        data = np.empty((height, width), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs="EPSG:3857",
            resampling=Resampling.nearest   # nearest for classified data
        )
        bounds_3857 = rasterio.transform.array_bounds(height, width, transform)
    return data, transform, bounds_3857   # bounds = (left, bottom, right, top)

# ── UNet flood prediction ──
flood_path = "outputs/unet_flood_prediction.tif"
flood_data, flood_transform, flood_bounds_3857 = reproject_to_3857(flood_path)
print(f"Flood loaded & reprojected: {flood_data.shape}")
print(f"  Bounds (3857): {[f'{v:.0f}' for v in flood_bounds_3857]}")

# Bounds tuple convenience
fl, fb, fr, ft = flood_bounds_3857       # left, bottom, right, top
xl, yl = (fl, fr), (fb, ft)             # xlim, ylim tuples

# imshow extent = [left, right, bottom, top]
map_extent = [fl, fr, fb, ft]

# ── LULC (reproject if available) ──
has_lulc, lulc_data, lulc_transform, lulc_extent = False, None, None, None
try:
    lulc_raw, lulc_tf, lulc_bnds = reproject_to_3857("data/vijayawada_lulc.tif")
    mn, mx = lulc_raw.min(), lulc_raw.max()
    lulc_data = (lulc_raw - mn)/(mx - mn) if mx != mn else np.ones_like(lulc_raw)
    lulc_transform = lulc_tf
    lulc_extent = [lulc_bnds[0], lulc_bnds[2], lulc_bnds[1], lulc_bnds[3]]
    has_lulc = True
    print(f"LULC loaded: {lulc_data.shape}")
except Exception as e:
    print(f"LULC not loaded: {e}")

# ── Roads — try both path locations (gis/ and data/) ──
has_roads, roads_major, roads_minor = False, None, None
for roads_path in ["gis/gis_osm_roads_free_1.shp",
                   "data/gis_osm_roads_free_1.shp"]:
    try:
        roads_gdf = gpd.read_file(roads_path).to_crs(3857)
        clip_box  = shapely_box(fl, fb, fr, ft)
        roads_gdf = roads_gdf.clip(clip_box)
        if not roads_gdf.empty:
            if 'fclass' in roads_gdf.columns:
                major_types = ['primary', 'trunk', 'motorway', 'secondary']
                roads_major = roads_gdf[roads_gdf['fclass'].isin(major_types)]
                roads_minor = roads_gdf[~roads_gdf['fclass'].isin(major_types)]
            else:
                roads_major = roads_gdf
                roads_minor = gpd.GeoDataFrame()
            has_roads = True
            print(f"Roads loaded from {roads_path}: {len(roads_gdf)} features "
                  f"({len(roads_major)} major, {len(roads_minor)} minor)")
            break
    except Exception as e:
        continue
if not has_roads:
    print("Roads not loaded — checked gis/ and data/ folders")

# ── Drainage — try both path locations ──
has_drainage, drainage_gdf = False, None
for drain_path in ["gis/gis_osm_waterways_free_1.shp",
                   "data/gis_osm_waterways_free_1.shp"]:
    try:
        drainage_gdf = gpd.read_file(drain_path).to_crs(3857)
        clip_box     = shapely_box(fl, fb, fr, ft)
        drainage_gdf = drainage_gdf.clip(clip_box)
        if not drainage_gdf.empty:
            has_drainage = True
            print(f"Drainage loaded from {drain_path}: {len(drainage_gdf)} features")
            break
    except Exception:
        continue
if not has_drainage:
    print("Drainage not loaded — checked gis/ and data/ folders")

# ── High-risk impact zone (buffer, same as CNN script) ──
flood_norm = flood_data.copy().astype(float)
flood_norm = np.nan_to_num(flood_norm)
if flood_norm.max() != flood_norm.min():
    flood_norm = (flood_norm - flood_norm.min()) / (flood_norm.max() - flood_norm.min())

p66 = np.percentile(flood_norm, 66)
high_mask = (flood_norm > p66).astype(np.uint8)

has_impact, impact_gdf = False, None
try:
    results = [
        {"properties": {"val": v}, "geometry": s}
        for s, v in shapes(high_mask, transform=flood_transform)
        if v == 1
    ]
    if results:
        impact_gdf = gpd.GeoDataFrame.from_features(results, crs="EPSG:3857")
        impact_gdf = gpd.GeoDataFrame(geometry=impact_gdf.buffer(80), crs="EPSG:3857")
        has_impact = True
        print(f"Impact zones computed: {len(impact_gdf)} polygons")
except Exception as e:
    print(f"Impact zone not computed: {e}")

# ============================================================
# 2. COLOR SCHEME  (soft, layered — avoids harsh full-sat red)
# ============================================================
risk_colors  = ['#a8c8e8',  '#f5c842',  '#e8645a']   # blue / amber / coral
class_names  = {0: 'Low Risk (Safe)', 1: 'Medium Risk (Caution)', 2: 'High Risk (Danger)'}
class_colors_map = {0: '#a8c8e8', 1: '#f5c842', 2: '#e8645a'}

cmap  = ListedColormap(risk_colors)
bnorm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

# ============================================================
# 3. FIGURE LAYOUT
#
#  ┌──────────────────────────────────┬──────────────┐
#  │                                  │              │
#  │           MAIN MAP               │  MAP LEGEND  │
#  │        (with basemap)            │    PANEL     │
#  ├─────────────────────┬────────────┴──────────────┤
#  │     BAR CHART       │       PIE CHART           │
#  ├─────────────────────┴───────────────────────────┤
#  │                 STATS BOX                       │
#  └─────────────────────────────────────────────────┘
# ============================================================
fig = plt.figure(figsize=(28, 20), dpi=100)
gs  = GridSpec(
    3, 2,
    figure=fig,
    width_ratios=[5.8, 1.3],
    height_ratios=[3.6, 1.4, 0.65],
    hspace=0.34,
    wspace=0.05,
    left=0.05, right=0.97,
    top=0.93,  bottom=0.05
)

ax_map   = fig.add_subplot(gs[0, 0])
ax_leg   = fig.add_subplot(gs[0, 1]);  ax_leg.axis('off')
ax_bar   = fig.add_subplot(gs[1, 0])
ax_pie   = fig.add_subplot(gs[1, 1])
ax_stats = fig.add_subplot(gs[2, :]);  ax_stats.axis('off')

# ============================================================
# 4. MAP — layer order matches CNN script approach
# ============================================================

# Set limits first so contextily knows the view extent
ax_map.set_xlim(*xl)
ax_map.set_ylim(*yl)

# --- Layer 0: CartoDB Positron basemap (same as CNN) ---
try:
    ctx.add_basemap(
        ax_map,
        source=ctx.providers.CartoDB.Positron,
        zoom=13,
        alpha=0.45,
        zorder=0
    )
    print("Basemap drawn")
except Exception as e:
    ax_map.set_facecolor('#e8eff7')
    print(f"Basemap not available (offline?): {e}")

# --- Layer 1: LULC faint texture ---
if has_lulc:
    ax_map.imshow(lulc_data, cmap='Greys', interpolation='bilinear',
                  extent=lulc_extent, aspect='auto',
                  zorder=1, alpha=0.15, vmin=0, vmax=1)
    print("LULC drawn")

# --- Layer 2: Flood risk (main raster, semi-transparent) ---
im_flood = ax_map.imshow(
    flood_data, cmap=cmap, norm=bnorm,
    interpolation='nearest', extent=map_extent,
    aspect='auto', zorder=2, alpha=0.75
)
ax_map.set_xlim(*xl);  ax_map.set_ylim(*yl)
print("Flood layer drawn")

# --- Layer 3: High-risk impact buffer (dark red halo) ---
if has_impact:
    impact_gdf.plot(ax=ax_map, color='#8b0000', alpha=0.18,
                    edgecolor='none', zorder=3)
    ax_map.set_xlim(*xl);  ax_map.set_ylim(*yl)
    print("Impact zones drawn")

# --- Layer 4: Minor roads ---
if has_roads and not roads_minor.empty:
    roads_minor.plot(ax=ax_map, color='#444444', linewidth=0.5,
                     alpha=0.45, zorder=4)
    ax_map.set_xlim(*xl);  ax_map.set_ylim(*yl)

# --- Layer 5: Major roads (bold, like CNN) ---
if has_roads and not roads_major.empty:
    roads_major.plot(ax=ax_map, color='#111111', linewidth=2.0,
                     alpha=0.85, zorder=5)
    ax_map.set_xlim(*xl);  ax_map.set_ylim(*yl)
    print("Roads drawn")

# --- Layer 6: Drainage / waterways (bright blue, on top) ---
if has_drainage:
    drainage_gdf.plot(ax=ax_map, color='#0066cc', linewidth=1.8,
                      alpha=0.95, zorder=6)
    ax_map.set_xlim(*xl);  ax_map.set_ylim(*yl)
    print("Drainage drawn")

# ── Map cosmetics ──
ax_map.set_title('Vijayawada Urban Flood Risk Map  (UNet Prediction)',
                 fontsize=15, fontweight='bold', pad=12)
ax_map.set_xlabel('Easting (m, EPSG:3857)', fontsize=9, labelpad=5)
ax_map.set_ylabel('Northing (m, EPSG:3857)', fontsize=9, labelpad=5)
ax_map.tick_params(labelsize=7.5)
ax_map.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1e6:.3f}M'))
ax_map.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y/1e6:.3f}M'))
ax_map.grid(True, color='white', linewidth=0.35, alpha=0.5, zorder=10)
for sp in ax_map.spines.values():
    sp.set_linewidth(1.2);  sp.set_edgecolor('#222')

# Colorbar
cbar = plt.colorbar(im_flood, ax=ax_map,
                    boundaries=[-0.5, 0.5, 1.5, 2.5], ticks=[0, 1, 2],
                    orientation='vertical', shrink=0.65, pad=0.012)
cbar.ax.set_yticklabels(['Low\n(Safe)', 'Medium\n(Caution)', 'High\n(Danger)'],
                         fontsize=8, fontweight='bold')
cbar.set_label('Flood Risk Level', fontsize=9, labelpad=8)

# Watermark
ax_map.text(0.99, 0.012,
            'Vijayawada Metropolitan Area  |  UNet Flood Risk Assessment',
            transform=ax_map.transAxes, fontsize=7.5,
            ha='right', va='bottom', style='italic', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff8dc',
                      alpha=0.92, edgecolor='#8B6914', linewidth=1.0),
            zorder=20)

# ============================================================
# 5. LEGEND PANEL — separate axis, never overlaps map
# ============================================================
legend_entries = [
    ('patch', '#a8c8e8', 'Low Risk Zone (Safe)'),
    ('patch', '#f5c842', 'Medium Risk Zone (Caution)'),
    ('patch', '#e8645a', 'High Risk Zone (Danger)'),
]
if has_impact:
    legend_entries.append(('patch', '#8b0000', 'High-Risk Impact Buffer'))
if has_drainage:
    legend_entries.append(('line',  '#0066cc', 'Rivers & Canals'))
if has_roads:
    legend_entries.append(('line',  '#111111', 'Major Roads'))
    legend_entries.append(('line',  '#666666', 'Minor Roads'))
if has_lulc:
    legend_entries.append(('patch', '#bbbbbb', 'Urban Dev. (LULC)'))

handles = []
for kind, color, label in legend_entries:
    if kind == 'patch':
        alpha = 0.35 if 'Buffer' in label else 0.85
        handles.append(mpatches.Patch(facecolor=color, edgecolor='#333',
                                       linewidth=1.0, alpha=alpha, label=label))
    else:
        lw = 2.2 if 'Major' in label else 1.3
        handles.append(mlines.Line2D([], [], color=color, linewidth=lw, label=label))

leg = ax_leg.legend(
    handles=handles,
    loc='upper left',
    bbox_to_anchor=(-0.08, 1.0),
    fontsize=9,
    title='MAP LEGEND',
    title_fontsize=10,
    framealpha=0.97,
    edgecolor='#333',
    fancybox=True,
    borderpad=1.0,
    labelspacing=0.9,
    handlelength=2.2,
)
leg.get_title().set_fontweight('bold')
leg.get_title().set_fontsize(11)

# ============================================================
# 6. BAR CHART
# ============================================================
unique_classes, counts = np.unique(flood_data, return_counts=True)
bar_labels = [class_names.get(int(c), f'Class {c}') for c in unique_classes]
bar_colors = [class_colors_map.get(int(c), '#888')   for c in unique_classes]

bars = ax_bar.bar(range(len(unique_classes)), counts,
                  color=bar_colors, edgecolor='#222', linewidth=1.2, width=0.5)
ax_bar.set_xlabel('Flood Risk Class', fontsize=10, fontweight='bold')
ax_bar.set_ylabel('Pixel Count',      fontsize=10, fontweight='bold')
ax_bar.set_title('Class Distribution', fontsize=12, fontweight='bold', pad=8)
ax_bar.set_xticks(range(len(unique_classes)))
ax_bar.set_xticklabels(bar_labels, fontsize=8.5)
ax_bar.grid(axis='y', alpha=0.22, linestyle='--', linewidth=0.7)
ax_bar.set_facecolor('#fafafa')
ax_bar.yaxis.set_major_formatter(
    FuncFormatter(lambda x, _: f'{int(x/1000)}K' if x >= 1000 else str(int(x))))
for bar, count in zip(bars, counts):
    pct = count / flood_data.size * 100
    ax_bar.text(bar.get_x() + bar.get_width()/2., bar.get_height()*0.50,
                f'{int(count):,}\n({pct:.1f}%)',
                ha='center', va='center', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          alpha=0.88, edgecolor='#555', linewidth=0.7))

# ============================================================
# 7. PIE CHART
# ============================================================
percentages = counts / flood_data.size * 100
explode     = [0.06 if p < 35 else 0.02 for p in percentages]
wedges, texts, autotexts = ax_pie.pie(
    percentages, labels=bar_labels, colors=bar_colors,
    autopct='%1.1f%%', startangle=90,
    textprops={'fontsize': 8.5, 'fontweight': 'bold'},
    explode=explode, shadow=False,
    wedgeprops={'edgecolor': '#222', 'linewidth': 1.2}
)
ax_pie.set_title('Risk % Distribution', fontsize=12, fontweight='bold', pad=8)
for at in autotexts:
    at.set_color('white');  at.set_fontweight('bold');  at.set_fontsize(9)

# ============================================================
# 8. STATS BOX
# ============================================================
counts_dict = dict(zip(unique_classes.tolist(), counts.tolist()))
low_risk    = counts_dict.get(0, 0)
medium_risk = counts_dict.get(1, 0)
high_risk   = counts_dict.get(2, 0)
total       = int(flood_data.size)
low_pct     = low_risk    / total * 100
medium_pct  = medium_risk / total * 100
high_pct    = high_risk   / total * 100

sep = '-' * 105
stats_text = '\n'.join([
    ' FLOOD RISK SUMMARY  (UNet Model — Vijayawada Urban Area)',
    sep,
    (f'  LOW RISK (Safe)        {low_risk:>12,} px  ({low_pct:5.1f}%)  |'
     f'  MEDIUM RISK (Caution) {medium_risk:>12,} px  ({medium_pct:5.1f}%)  |'
     f'  HIGH RISK (Danger)  {high_risk:>12,} px  ({high_pct:5.1f}%)'),
    sep,
    (f'  Total: {total:,} px  |  Size: {flood_data.shape[0]}x{flood_data.shape[1]}  |  '
     f'CRS: EPSG:3857  |  '
     f'Drainage: {"Available" if has_drainage else "N/A"}  |  '
     f'Roads: {"Available" if has_roads else "N/A"}  |  '
     f'LULC: {"Available" if has_lulc else "N/A"}')
])
ax_stats.text(0.5, 0.5, stats_text,
              transform=ax_stats.transAxes,
              fontsize=8.5, ha='center', va='center',
              family='monospace',
              bbox=dict(boxstyle='round,pad=0.8', facecolor='#f2f2f2',
                        edgecolor='#333', linewidth=1.4, alpha=0.97))

# ============================================================
# 9. MAIN TITLE & SAVE
# ============================================================
fig.suptitle('VIJAYAWADA URBAN FLOOD RISK & DRAINAGE ANALYSIS DASHBOARD',
             fontsize=17, fontweight='bold', y=0.975)

os.makedirs('outputs', exist_ok=True)
out_path = 'outputs/flood_analysis_dashboard.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nDashboard saved -> {out_path}")

try:
    plt.show()
except KeyboardInterrupt:
    pass
finally:
    plt.close('all')

# ============================================================
# 10. CONSOLE REPORT
# ============================================================
print("\n" + "="*80)
print("  VIJAYAWADA FLOOD RISK REPORT  (UNet)")
print("="*80)
print(f"  Image : {flood_data.shape[0]} x {flood_data.shape[1]} px  |  Total: {total:,}")
print(f"  Bounds (3857): L={fl:.0f}  B={fb:.0f}  R={fr:.0f}  T={ft:.0f}\n")
print(f"  {'Category':<28} {'Pixels':>14} {'Pct':>10}  Status")
print("  " + "-"*65)
for cat, px, pct, st in [
    ('Low Risk (Safe)',        low_risk,    low_pct,    'Safe Zone'),
    ('Medium Risk (Caution)',  medium_risk, medium_pct, 'Warning Zone'),
    ('High Risk (Danger)',     high_risk,   high_pct,   'Critical Zone'),
]:
    print(f"  {cat:<28} {px:>14,} {pct:>9.2f}%  {st}")
print("  " + "-"*65)
print(f"  {'TOTAL':<28} {total:>14,}    100.00%")
print("="*80 + "\n")