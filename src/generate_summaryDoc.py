import numpy as np 
import geopandas as gpd
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages
from shapely.geometry import shape
import os 
import glob 
from pyproj import Transformer
import pdb 
from shapely.wkt import loads as wkt_loads
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr 
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd 
from matplotlib.gridspec import GridSpec
import sys 
from matplotlib.colors import ListedColormap, BoundaryNorm

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

#homebrewed
import fireEventTracking as FET


def subset_from_ds(ds, varname,  x_center, y_center, time_sub, half_width_km=25):
    deg_per_km = 1 / 111.0  # Approximate for lat/lon (valid near equator)

    # --- Load dataset ---
    lon = ds['lon']
    lat = ds['lat']
    data = ds[varname]  # Shape: (time, lat, lon)

    '''
    if varname == 'MODEL_FIRE':
        # Select latest time step (e.g., last)
        # Step 1: Extract date part as integer
        yyyymmdd = int(time_sub.strftime('%Y%m%d'))

        # Step 2: Compute fractional part of day
        seconds_in_day = 24 * 60 * 60
        seconds_since_midnight = (
            time_sub.hour * 3600 +
            time_sub.minute * 60 +
            time_sub.second +
            time_sub.microsecond / 1e6
        )
        fraction = seconds_since_midnight / seconds_in_day

        # Step 3: Combine
        time_sub_float = yyyymmdd + fraction
    else: 
        time_sub_float = time_sub
    '''
    
    fire_slice = data.sel(time=time_sub, method='nearest')
    
    # --- Subset by lat/lon bounds ---
    lon_min = x_center - half_width_km * deg_per_km
    lon_max = x_center + half_width_km * deg_per_km
    lat_min = y_center - half_width_km * deg_per_km
    lat_max = y_center + half_width_km * deg_per_km

    subset = fire_slice.sel(
        lon=slice(lon_min, lon_max),
        lat=slice(lat_min, lat_max)  # Note: descending if lat is decreasing
    )

    return subset, lon_min, lon_max, lat_min, lat_max


def decimal_to_dms(decimal_degree):
    is_positive = decimal_degree >= 0
    decimal_degree = abs(decimal_degree)
    degrees = int(decimal_degree)
    minutes_full = (decimal_degree - degrees) * 60
    minutes = int(minutes_full)
    seconds = round((minutes_full - minutes) * 60, 2)

    if not is_positive:
        degrees = -degrees

    return degrees, minutes, seconds

inputName ='SILEX-MF'
sensorName = 'FCI'
log_dir = '/home/paugamr/Src/FireEventTracking/log_doc/'

params = FET.init(inputName,sensorName,log_dir)

dir_report = params['event']['dir_data']+'/Report'
os.makedirs(dir_report, exist_ok=True)

geojsons = sorted(glob.glob(params['event']['dir_geoJson']+'/*.geojson'))
last_geojson = geojsons[-1]
time_last_geojson = os.path.basename(last_geojson).split('.geojs')[0].split('ents-')[1]
time_report = pd.to_datetime(time_last_geojson, format='%Y-%m-%d_%H%M')

# Load the GeoJSON file
gdf = gpd.read_file(last_geojson)

# Filter French fires by 'name' field
gdf_france = gdf[gdf["name"].str.contains("_FR_", na=False)].copy()

# Estimate area in km² using EPSG:3857 projection
gdf_france = gdf_france.to_crs(epsg=3857)
gdf_france["area_km2"] = gdf_france.geometry.area / 1e6  # convert m² to km²


dir_pof = '/home/paugamr/data/POF' 
time_report.strftime('%Y%m%d')
file_latest_pof = f"{dir_pof}/POF_1KM_MF_{(time_report-pd.Timedelta(days=1)).strftime('%Y%m%d')}_FC.nc"
ds_pof = xr.open_dataset(file_latest_pof)

# Your color list
cols = ['#8ba9b3', '#edcc00', '#edcc00', '#e57d0f', '#e57d0f', '#e57d0f',
        '#e57d0f', '#e57d0f', '#e57d0f', '#e21819', '#e21819', '#e21819',
        '#e21819', '#e21819', '#000000', '#000000']
# Your levels
levels = np.arange(17) / 2500  # 17 boundaries define 16 intervals
# Create colormap and norm
cmap = ListedColormap(cols)
norm = BoundaryNorm(levels, ncolors=cmap.N)


# Sort by FRP descending and keep top 10
top_fires = gdf_france.sort_values("frp", ascending=False).head(10)
transformer = Transformer.from_crs("EPSG:{:d}".format(params['general']['crs']), "EPSG:4326", always_xy=True)

# Create PDF
with PdfPages(f"{dir_report}/fires_FR_{time_last_geojson}.pdf") as pdf:

    # Page 1: Summary Table
    fig = plt.figure(figsize=(8, 11))  # A4 landscape
    gs = GridSpec(2, 2, figure=fig)


    # Table
    ax_table = fig.add_subplot(gs[0, : ])
    ax_table.axis('off')

    table_data = [["Name", "Last Obs", "FRP (MW)", "Area (km²)"]]
    for _, row in top_fires.iterrows():
        table_data.append([
            ' '.join(row["name"].split('_')[2:4]),
            row["time"].strftime('%Y-%m-%d %H:%S'),
            f"{row['frp']:.2f}",
            f"{row['area_km2']:.2f}"
        ])

    table = ax_table.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc='left',
        loc='center'
    )

    table.auto_set_font_size(False)
    col_widths = [0.45, 0.3, 0.15, 0.2]
    n_rows = len(table_data)
    n_cols = len(table_data[0])

    for row in range(n_rows):
        for col in range(n_cols):
            cell = table[(row, col)]
            cell.set_width(col_widths[col])
            if row == 0:
                cell.set_fontsize(10)
                cell.set_text_props(weight='bold')
            else:
                cell.set_fontsize(9)

    table.scale(0.9, 1.4)

# Active fires info
    ax_table.text(
        0.2, -0.1,
        f"Number of active fires in France: {len(gdf_france)}\n"
        "(A fire is considered active in the current version of FET\n"
        "if it has at least one observation within the last 2 days)",
        fontsize=10, ha='left'
    )
    ax_table.text(0.5,0.9,
        f"Top 10 Fire Events in France by FRP\nwithin the last 30min time window processed\nstarting at {time_last_geojson}",
        fontsize=14, weight='bold', ha='center'
    )

    # Map
    loc_fire_top = []
    for _, row in top_fires.iterrows():
        pt = wkt_loads(row.center)
        x, y = pt.x, pt.y
        lon, lat = transformer.transform(x,y)
        loc_fire_top.append([lon,lat])
    loc_fire_top = np.array(loc_fire_top)
    loc_fire_all = []
    for _, row in gdf_france.iterrows():
        pt = wkt_loads(row.center)
        x, y = pt.x, pt.y
        lon, lat = transformer.transform(x,y)
        loc_fire_all.append([lon,lat])
    loc_fire_all = np.array(loc_fire_all)

    ax_map = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
    ax_map.set_extent([-6, 10.5, 41, 51.5], crs=ccrs.PlateCarree())
    ax_map.add_feature(cfeature.LAND, facecolor='lightgray')
    ax_map.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax_map.add_feature(cfeature.COASTLINE)
    ax_map.add_feature(cfeature.BORDERS, linestyle=':')
    ax_map.scatter(loc_fire_top[:,0], loc_fire_top[:,1], color='red', s=10, label='top10')
    ax_map.scatter(loc_fire_all[:,0], loc_fire_all[:,1], color='red', s=10, label='all active', alpha=0.5)
    plt.legend()
    
    ax_map = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
    ax_map.set_extent([-6, 10.5, 41, 51.5], crs=ccrs.PlateCarree())
    #ax_map.add_feature(cfeature.LAND, facecolor='lightgray')
    #ax_map.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax_map.add_feature(cfeature.COASTLINE)
    ax_map.add_feature(cfeature.BORDERS, linestyle=':')

    pof_now = ds_pof.sel(time=time_report,method='nearest')['MODEL_FIRE']
    pof_now.plot(ax=ax_map, cmap=cmap, norm=norm)
    ax_map.set_title(f'POF date={pof_now.time}')
    pdf.savefig(fig)
    plt.close(fig)


    for _, row in top_fires.iterrows():
        name = row["name"]
        frp = row["frp"]
        area = row["area_km2"]
        time = row["time"]
        image_url = row["image"]  # This is the FRP time series image
        
        pt = wkt_loads(row.center)
        x, y = pt.x, pt.y
        lon, lat = transformer.transform(x,y)

        print(name)

        # Load FRP time series image
        try:
            response = requests.get(image_url,verify=False)
            response.raise_for_status()
            frp_img = Image.open(BytesIO(response.content))
        except Exception as e:
            print(f"FRP image for {name} not available: {e}")
            continue

        fig = plt.figure(figsize=(8, 11))
        fig.suptitle(name, fontsize=14, weight='bold', y=0.98)

        # === 1. FRP Time Series (as image) ===
        ax_frp = fig.add_axes([0.15, 0.73, 0.8, 0.2])  # [left, bottom, width, height]
        ax_frp.imshow(frp_img)
        ax_frp.axis('off')
        ax_frp.set_title("FRP Time Series", fontsize=10)
        
        # === 1.1. cart location fire ===
        ax_loc = fig.add_axes([0.05, 0.71, 0.25, 0.2], projection=ccrs.PlateCarree())
        ax_loc.scatter(lon,lat, s=5, color='r')
        #ax_loc.axis('off')
        # Base map
        ax_loc.add_feature(cfeature.COASTLINE)
        ax_loc.add_feature(cfeature.BORDERS, linestyle=':')
        ax_loc.add_feature(cfeature.LAND, facecolor='lightgray')
        ax_loc.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax_loc.set_extent([-6, 10.5, 41, 51.5], crs=ccrs.PlateCarree())
        lat_dms = decimal_to_dms(lat)
        lon_dms = decimal_to_dms(lon)
        ax_loc.set_title(f"lon: {lon_dms[0]}°{lon_dms[1]}'{lon_dms[2]}\"\n"+f"lat: {lat_dms[0]}°{lat_dms[1]}'{lat_dms[2]}\"" ) 
        
        # === 2. 2×2 Grid of Square imshow Panels ===
        imshow_size = 0.3  # square size for width and height

        positions = [
            [0.1, 0.40], [0.6, 0.40],   # Top row (more horizontal spacing)
            [0.1, 0.15], [0.6, 0.15]    # Bottom row (more vertical spacing)
        ]
        title_panel = ['POF', 'RGB', 'IR38', 'IR22']
        varname_panel = ['MODEL_FIRE', '', '', '' ]
        title_panel = ['POF', '', '', '' ]
        cmap_panel = ['viridis', 'viridis', 'inferno', 'jet']
        for i, (left, bottom) in enumerate(positions):
            
            if varname_panel[i] == '': continue
            
            if title_panel[i]=='POF':
                subset, lon_min, lon_max, lat_min, lat_max = subset_from_ds(ds_pof, varname_panel[i], lon, lat, time_report, half_width_km=25)
            else:
                subset = np.random.rand(10, 10)
            
            '''
            ax = fig.add_axes([left, bottom, imshow_size, imshow_size])
            ax.imshow(dummy_array, cmap='viridis')
            ax.axis('off')
            ax.set_title(title_panel[i], fontsize=9)
            '''

            ax = fig.add_axes([left, bottom, imshow_size, imshow_size], projection=ccrs.PlateCarree())

            # Base map
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            ax.add_feature(cfeature.LAND, facecolor='lightgray')
            ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

            # Plot fire data
            pcm = ax.pcolormesh(
                subset['lon'], subset['lat'], subset,
                transform=ccrs.PlateCarree(),
                cmap=cmap_panel[i], shading='auto'
            )


            # Assuming ax is a GeoAxes from Cartopy
            cax = inset_axes(ax,
                             width="5%",  # width = 5% of parent_bbox width
                             height="100%",  # height = 100%
                             loc='lower left',
                             bbox_to_anchor=(-0.1, 0., 1, 1),
                             bbox_transform=ax.transAxes,
                             borderpad=0)

            # Now plot colorbar here
            plt.colorbar(pcm, cax=cax, orientation='vertical')   

            # Optional: move colorbar ticks to left side
            cax.yaxis.set_ticks_position('left')
            cax.yaxis.set_label_position('left')
            #(\nCentered at ({lon:.4f}, {lat:.4f})')
            ax.set_title(f"{title_panel[i]} t={pd.to_datetime(subset.time.values).strftime('%Y%m%dT%H:%MZ')}") 


        # === 3. Metadata ===
        metadata = f"Time (UTC): {time}\nFRP: {frp:.2f} MW\nArea: {area:.2f} km²"
        ax_text = fig.add_axes([0.1, 0.01, 0.8, 0.1])
        ax_text.axis('off')
        ax_text.text(0.5, 0.5, metadata, ha='center', va='center', fontsize=10)

        pdf.savefig(fig)
        plt.close(fig)

    '''
    # One page per fire
    for _, row in top_fires.iterrows():
        name = row["name"]
        frp = row["frp"]
        area = row["area_km2"]
        time = row["time"]
        image_url = row["image"]

        # Download fire image
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
        except Exception as e:
            print(f"Image for {name} not available: {e}")
            continue

        # Set up a layout with two axes: one for title/metadata, one for image
        fig = plt.figure(figsize=(8, 10))
        fig.suptitle(name, fontsize=14, weight='bold', y=0.95)

        # Metadata text
        metadata = f"Time (UTC): {time}\nFRP: {frp:.2f} MW\nArea: {area:.2f} km²"
        ax_text = fig.add_axes([0.1, 0.05, 0.8, 0.15])
        ax_text.axis('off')
        ax_text.text(0.5, 0.5, metadata, ha='center', va='center', fontsize=10)

        # Image display
        ax_img = fig.add_axes([0.1, 0.25, 0.8, 0.65])  # [left, bottom, width, height]
        ax_img.axis('off')
        ax_img.imshow(img)
        ax_img.set_title("FRP time series", fontsize=10)

        # Save to PDF
        pdf.savefig(fig)
        plt.close(fig)
    '''


print(f"✅ PDF created: {dir_report}/fires_FR_{time_last_geojson}.pdf")
