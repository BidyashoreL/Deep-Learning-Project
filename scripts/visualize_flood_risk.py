import rasterio
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the real data you just downloaded
with rasterio.open('vijayawada_dem.tif') as src:
    dem = src.read(1) # Read the first band (elevation)
    
    # Replace any error/no-data values with the minimum elevation
    dem[dem < -100] = np.min(dem[dem > -100])

# 2. Calculate the Slope (Gradient)
# This is a key feature for your CNN (12.4% factor)
dy, dx = np.gradient(dem)
slope = np.sqrt(dx**2 + dy**2)

# 3. Plotting the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

# Plot Elevation
im1 = ax1.imshow(dem, cmap='terrain')
ax1.set_title("Vijayawada Elevation (m)")
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

# Plot Slope (High risk areas are usually the flattest)
im2 = ax2.imshow(slope, cmap='magma', vmax=5) # Vmax 5 to highlight subtle urban slopes
ax2.set_title("Slope Analysis (Flood Susceptibility)")
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

print(f"Max Elevation: {np.max(dem)}m")
print(f"Min Elevation: {np.min(dem)}m")