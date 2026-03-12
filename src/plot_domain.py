import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import rcParams

# --- Enable LaTeX rendering ---
rcParams["text.usetex"] = True
rcParams["font.family"] = "serif"

# Define domains: (min_lon, min_lat, max_lon, max_lat)
domains = {
    r"\textbf{SILEX}": (-10, 35, 20, 52),
    r"\textbf{MED}": (-10, 29, 41, 47.5),
    r"\textbf{PORTUGAL}": (-10, 36, -5, 44),
}

# Create figure and axis
fig = plt.figure(figsize=(10, 5))
ax = plt.axes(projection=ccrs.PlateCarree())

# Set global extent covering all domains
ax.set_extent([-11.5, 42.5, 28, 53], crs=ccrs.PlateCarree())

# Add map features
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='white')

# Draw domains
for name, (min_lon, min_lat, max_lon, max_lat) in domains.items():
    
    # Draw rectangle
    ax.plot(
        [min_lon, max_lon, max_lon, min_lon, min_lon],
        [min_lat, min_lat, max_lat, max_lat, min_lat],
        transform=ccrs.PlateCarree(),
        linewidth=2,
    )

    # Add label at bottom-right corner
    ax.text(
        max_lon,
        min_lat,
        f"${name}$",
        transform=ccrs.PlateCarree(),
        fontsize=11,
        ha='right',
        va='bottom',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

# Gridlines
gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
gl.top_labels = False
gl.right_labels = False

plt.title(r"\textbf{Domain Configuration}")
plt.tight_layout()
fig.savefig('configDomainMap.png', dpi=300)
plt.close(fig)
