import numpy as np 
import matplotlib as mpl
mpl.use('Agg')  # or 'QtAgg'
import matplotlib.pyplot as plt
import geopandas as gpd
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
from pyproj import Geod
import xarray as xr 
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd 
from matplotlib.gridspec import GridSpec
import sys 
from matplotlib.colors import ListedColormap, BoundaryNorm
import subprocess
import math
import tempfile
import shutil
import rasterio
from rasterio.plot import show
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
import pandas as pd


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


######################################################
def generate_report(params,XX_TABLE_HERE, XX_NUMBER_OF_FIRE,time_report,firePages):

    with open(params['general']['srcDir']+'/fireReport/template.tex', 'rb') as f:
        lines = f.readlines()

    XX_TIME_REPORT_beg = time_report.strftime('%Y%m%dT %HH:%MMZ')
    XX_TIME_REPORT_end = (time_report+pd.Timedelta(minutes=30)).strftime('%Y%m%dT %HH:%MMZ')

    for ii, line in enumerate(lines):
        decoded_line = line.decode('utf-8')
        if 'XX_TABLE_HERE' in decoded_line:
            decoded_line = decoded_line.replace('XX_TABLE_HERE', XX_TABLE_HERE)
            lines[ii] = decoded_line.encode('utf-8')  # store back as bytes
        if 'XX_NUMBER_OF_FIRE' in decoded_line:
            decoded_line = decoded_line.replace('XX_NUMBER_OF_FIRE', XX_NUMBER_OF_FIRE)
            lines[ii] = decoded_line.encode('utf-8')
        if 'XX_TIME_REPORT_beg' in decoded_line:
            decoded_line = decoded_line.replace('XX_TIME_REPORT_beg', XX_TIME_REPORT_beg)
            lines[ii] = decoded_line.encode('utf-8')  # store back as bytes
        if 'XX_TIME_REPORT_end' in decoded_line:
            decoded_line = decoded_line.replace('XX_TIME_REPORT_end', XX_TIME_REPORT_end)
            lines[ii] = decoded_line.encode('utf-8')  # store back as bytes

    with open(f"{params['general']['reportDir']}/fireReport.tex", 'wb') as f:
        f.writelines(lines[:-1] + [item for sublist in firePages for item in sublist] + [lines[-1]])

    # Run pdflatex in that directory
    subprocess.run(
        ['pdflatex', 'fireReport.tex'],
        cwd=f"{params['general']['reportDir']}",
        check=True
    )
    subprocess.run(
        ['pdflatex', 'fireReport.tex'],
        cwd=f"{params['general']['reportDir']}",
        check=True
    )

######################################################
def firePage(params,XX_FIRE_NAME, XX_FIRE_VEG_MAP, XX_FIRE_FRP, XX_META_DATA, XX_FIRE_ID, XX_FFMNH_URL):

    with open(params['general']['srcDir']+'/fireReport/template_fire.tex', 'rb') as f:
        lines = f.readlines()

    for ii, line in enumerate(lines):
        decoded_line = line.decode('utf-8')
        if 'XX_FIRE_NAME' in decoded_line:
            decoded_line = decoded_line.replace('XX_FIRE_NAME', XX_FIRE_NAME)
            lines[ii] = decoded_line.encode('utf-8')  # store back as bytes
        if 'XX_FIRE_VEG_MAP' in decoded_line:
            decoded_line = decoded_line.replace('XX_FIRE_VEG_MAP', XX_FIRE_VEG_MAP)
            lines[ii] = decoded_line.encode('utf-8')
        if 'XX_FIRE_FRP' in decoded_line:
            decoded_line = decoded_line.replace('XX_FIRE_FRP', XX_FIRE_FRP)
            lines[ii] = decoded_line.encode('utf-8')  # store back as bytes
        if 'XX_META_DATA' in decoded_line:
            decoded_line = decoded_line.replace('XX_META_DATA', XX_META_DATA)
            lines[ii] = decoded_line.encode('utf-8')  # store back as bytes
        if 'XX_FIRE_ID' in decoded_line:
            decoded_line = decoded_line.replace('XX_FIRE_ID', XX_FIRE_ID)
            lines[ii] = decoded_line.encode('utf-8')  # store back as bytes
        if 'XX_FFMNH_URL' in decoded_line:
            if XX_FFMNH_URL == 'none': np.delete(lines,ii)
            decoded_line = decoded_line.replace('XX_FFMNH_URL', XX_FFMNH_URL)
            lines[ii] = decoded_line.encode('utf-8')  # store back as bytes

    return lines


################################
def bounding_box_km(lon, lat, half_side_km=20):
    # Earth radius (WGS-84)
    R = 6371.0  # km

    # Latitude: ~111.32 km per degree
    dlat = half_side_km / 111.32

    # Longitude degrees per km depends on latitude
    dlon = half_side_km / (111.32 * math.cos(math.radians(lat)))

    min_lon = lon - dlon
    max_lon = lon + dlon
    min_lat = lat - dlat
    max_lat = lat + dlat

    return [min_lon, max_lon, min_lat, max_lat]


################################
def plot_pof(params,pof_now, filename, extent=[-6, 10.5, 41, 51.5], fireloc=(0,0), flag_colorbar=False, flag_scalebar=False):

    jid = int(filename.split('.')[0][-1])

    mpl.rcdefaults()
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['axes.linewidth'] = 1
    mpl.rcParams['axes.labelsize'] = 12.
    mpl.rcParams['legend.fontsize'] = 'small'
    mpl.rcParams['legend.fancybox'] = True
    mpl.rcParams['font.size'] = 12.
    mpl.rcParams['xtick.labelsize'] = 12.
    mpl.rcParams['ytick.labelsize'] = 12.
    mpl.rcParams['figure.subplot.left'] = .05
    mpl.rcParams['figure.subplot.right'] = .95
    mpl.rcParams['figure.subplot.top'] = .95
    mpl.rcParams['figure.subplot.bottom'] = .05
    mpl.rcParams['figure.subplot.hspace'] = 0.05
    mpl.rcParams['figure.subplot.wspace'] = 0.05  
    fig = plt.figure(figsize=(8,6))
    ax_map = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax_map.set_extent(extent, crs=ccrs.PlateCarree())
    #ax_map.add_feature(cfeature.LAND, facecolor='lightgray')
    #ax_map.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax_map.add_feature(cfeature.COASTLINE)
    ax_map.add_feature(cfeature.BORDERS, linestyle=':')

    ax_map.pcolormesh(pof_now.lon, pof_now.lat, pof_now, cmap=cmap, norm=norm)
    ax_map.scatter(fireloc[0],fireloc[1], facecolor='none',edgecolor='k',s=100, linewidth=2  )
    ax_map.set_title(f'POF date={pof_now.time.values} (j+{jid})',fontsize=20)

    if flag_scalebar:
        add_scalebar(ax_map, 2, location=(0.1, 0.05), linewidth=3, text_offset=0.01)

    fig.savefig(f"{params['general']['reportDir']}/{filename}")
    plt.close(fig)

    if flag_colorbar:
        # Create a separate figure for colorbar
        fig_cb = plt.figure(figsize=(1.2, 6))  # adjust size as needed
        ax_cb = fig_cb.add_axes([0.0, 0.1, 0.2, 0.73])  # [left, bottom, width, height]

        # Create a dummy mappable with your colormap and norm
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        # Draw the colorbar
        cbar = fig_cb.colorbar(sm, cax=ax_cb, orientation='vertical')
        cbar.set_label('POF', fontsize=12)  # label as needed
        cbar.ax.tick_params(labelsize=10)

        fig_cb.savefig(f"{params['general']['reportDir']}/{filename.split('.')[0]}_colorbar.png", dpi=100, bbox_inches='tight')
        plt.close(fig_cb)


############################
def add_scalebar(ax, length_km, location=(0.1, 0.05), linewidth=3, text_offset=0.01):
    """
    Add a scale bar to a Cartopy map.

    Parameters:
    - ax: matplotlib Axes with Cartopy projection
    - length_km: scale bar length in kilometers
    - location: tuple (x, y) in axis fraction coordinates (0–1)
    - linewidth: thickness of the scale line
    - text_offset: vertical offset of text label (in axes fraction)
    """
    # Get extent in data coordinates (lon/lat)
    lon_min, lon_max, lat_min, lat_max = ax.get_extent(ccrs.PlateCarree())

    # Convert axis fraction coords to data coords
    lon_start = lon_min + location[0] * (lon_max - lon_min)
    lat_start = lat_min + location[1] * (lat_max - lat_min)

    # Compute destination point at given distance using pyproj.Geod
    geod = Geod(ellps="WGS84")
    lon_end, lat_end, _ = geod.fwd(lon_start, lat_start, 90, length_km * 1000)  # azimuth 90° = east

    # Plot scale bar
    ax.plot([lon_start, lon_end], [lat_start, lat_start],
            transform=ccrs.PlateCarree(), color='k', linewidth=linewidth, solid_capstyle='butt')

    # Add label
    ax.text((lon_start + lon_end) / 2, lat_start + text_offset * (lat_max - lat_min),
            f'{length_km} km', transform=ccrs.PlateCarree(),
            ha='center', va='bottom', fontsize=20)
############################
if __name__ == '__main__':
############################ 
    inputName ='SILEX-MF'
    sensorName = 'FCI'
    log_dir = os.path.dirname(os.path.abspath(__file__)) +'/../log_doc/'

    params = FET.init(inputName,sensorName,log_dir)
    dir_report = params['event']['dir_data']+'/Report'
    os.makedirs(dir_report, exist_ok=True)
    params['general']['srcDir']= os.path.dirname(__file__)
    tmpDir = tempfile.TemporaryDirectory()
    params['general']['reportDir']= tmpDir.name
    shutil.copy( f"{params['general']['srcDir']}/fireReport/logo_silex-2-06d25.png",f"{params['general']['reportDir']}/logo_silex-2-06d25.png")
    

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


    #POF
    #####
    dir_pof = f"{params['general']['root_data']}/POF"
    XX_TIME_REPORT = time_report.strftime('%Y%m%dT %HH:%MMZ')
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

    # CLC colorcode
    legend_path = f"{params['general']['root_data']}/CorineLandCover/u2018_clc2018_v2020_20u1_raster100m/Legend/CLC2018_CLC2018_V2018_20_QGIS.txt"
    legend_df = pd.read_csv(
        legend_path,
        header=None,
        names=['code', 'r', 'g', 'b', 'a', 'label']
    )
    # Create hex color from RGB
    legend_df['color'] = legend_df.apply(
        lambda row: "#{:02x}{:02x}{:02x}".format(row['r'], row['g'], row['b']),
        axis=1
    )
    legend_df['code'] = legend_df['code'].astype(int)

    # Sort by FRP descending and keep top 10
    top_fires = gdf_france.sort_values("frp", ascending=False).head(10)
    transformer = Transformer.from_crs("EPSG:{:d}".format(params['general']['crs']), "EPSG:4326", always_xy=True)

    # Create PDF
    
    XX_TABLE_HERE = ''
    for _, row in top_fires.iterrows():
        XX_TABLE_HERE +=\
            f"\hyperref[sec:fireID{row['id_fire_event']}]"+"{" + ' '.join(row["name"].split('_')[2:4]) + '}' + '&' +\
            row["time"].strftime('%Y-%m-%d %H:%S')+ '&' +\
            f"{row['frp']:.2f}"+ '&' +\
            f"{row['area_km2']:.2f}" + '\\\\'
        XX_TABLE_HERE += '\n'

    XX_NUMBER_OF_FIRE = f'{len(gdf_france)}'

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
    
    mpl.rcdefaults()
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['axes.linewidth'] = 1
    mpl.rcParams['axes.labelsize'] = 12.
    mpl.rcParams['legend.fontsize'] = 'small'
    mpl.rcParams['legend.fancybox'] = True
    mpl.rcParams['font.size'] = 12.
    mpl.rcParams['xtick.labelsize'] = 12.
    mpl.rcParams['ytick.labelsize'] = 12.
    mpl.rcParams['figure.subplot.left'] = .05
    mpl.rcParams['figure.subplot.right'] = .95
    mpl.rcParams['figure.subplot.top'] = .95
    mpl.rcParams['figure.subplot.bottom'] = .05
    mpl.rcParams['figure.subplot.hspace'] = 0.05
    mpl.rcParams['figure.subplot.wspace'] = 0.05  
    fig = plt.figure(figsize=(8,6))
    ax_map = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax_map.set_extent([-6, 10.5, 41, 51.5], crs=ccrs.PlateCarree())
    ax_map.add_feature(cfeature.LAND, facecolor='lightgray')
    ax_map.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax_map.add_feature(cfeature.COASTLINE)
    ax_map.add_feature(cfeature.BORDERS, linestyle=':')
    ax_map.scatter(loc_fire_all[:,0], loc_fire_all[:,1], color='blue', s=30, label='other active' )
    ax_map.scatter(loc_fire_top[:,0], loc_fire_top[:,1], color='red', s=30, label='top10')
    ax_map.set_title(f'all active fire date={time_report}',fontsize=20)
    plt.legend()
             
    fig.savefig(f"{params['general']['reportDir']}/general_scatterPlot.png")
    plt.close(fig)
   
    extent=[-6, 10.5, 41, 51.5] 

    pof_now = ds_pof.sel(time=time_report,method='nearest')['MODEL_FIRE']
    plot_pof(params,pof_now, 'general_pof_j0.png',flag_colorbar=True)
    
    pof_j1 = ds_pof.sel(time=time_report+pd.Timedelta(days=1),method='nearest')['MODEL_FIRE']
    plot_pof(params,pof_j1, 'general_pof_j1.png')
    pof_j2 = ds_pof.sel(time=time_report+pd.Timedelta(days=2),method='nearest')['MODEL_FIRE']
    plot_pof(params,pof_j2, 'general_pof_j2.png')
    pof_j3 = ds_pof.sel(time=time_report+pd.Timedelta(days=3),method='nearest')['MODEL_FIRE']
    plot_pof(params,pof_j3, 'general_pof_j3.png')


    linesFires = []
    #
    # loop over top10 fire
    #
    for _, row in top_fires.iterrows():
        name = row["name"]
        frp = row["frp"]
        area = row["area_km2"]
        time = row["time"]
        image_url = row["image"]  # This is the FRP time series image
        ffmnh_url = row["ffmnhUrl"]    

        pt = wkt_loads(row.center)
        x, y = pt.x, pt.y
        lon, lat = transformer.transform(x,y)

        print(name)

        # Load FRP time series image
        #try:
        frp_img_path = f"{params['general']['root_data']}/{sensorName}/{'/'.join(image_url.split('/')[-3:])}"
        frp_img = Image.open(frp_img_path) 
        #response = requests.get(image_url,verify=False)
        #response.raise_for_status()
        #frp_img = Image.open(BytesIO(response.content))
        XX_FIRE_FRP = f"frp{row['id_fire_event']}.png"
        frp_img.save(f"{params['general']['reportDir']}/{XX_FIRE_FRP}")
        #except Exception as e:
        #    print(f"FRP image for {name} not available: {e}")
        #    XX_FIRE_FRP=''
        


        XX_FIRE_NAME = ' '.join(name.split('_')[2:4])
        XX_FIRE_LOC_MAP = f'loc{row["id_fire_event"]}.png'
        XX_FIRE_VEG_MAP = f'veg{row["id_fire_event"]}.png'

        XX_FFMNH_URL = ffmnh_url


        # === 1.1. cart vegetation around fire ===
        with rasterio.open(f"{params['general']['root_data']}/CorineLandCover/u2018_clc2018_v2020_20u1_raster100m/DATA/U2018_CLC2018_V2020_20u1.tif") as src:
            # Reproject bounding box to raster CRS
            lon_min, lon_max, lat_min, lat_max = bounding_box_km(lon, lat, half_side_km=5)
            bbox_wgs84 = (lon_min, lat_min, lon_max, lat_max)
            bbox_proj = transform_bounds("EPSG:4326", src.crs, *bbox_wgs84)
            
            # Compute window in raster coordinates
            window = from_bounds(*bbox_proj, transform=src.transform)
            
            # Read data within window
            vegetation_crop = src.read(1, window=window, resampling=Resampling.nearest)
            
            # Get updated transform for cropped array
            transform_crop = src.window_transform(window)
       
        codes_in_crop = np.unique(vegetation_crop)
        legend_filtered = legend_df[legend_df.index.isin(codes_in_crop)].sort_values("code")

        cmap_veg = ListedColormap(legend_filtered['color'].values)
        values_colorbar = np.append(legend_filtered.index, legend_filtered.index[-1] + 1)
        norm_veg = BoundaryNorm(
            values_colorbar,
            cmap_veg.N
        ) 

        # Set up the plot with PlateCarree (WGS84)
        mpl.rcParams['text.usetex'] = True
        mpl.rcParams['axes.linewidth'] = 1
        mpl.rcParams['axes.labelsize'] = 12.
        mpl.rcParams['legend.fontsize'] = 20.
        mpl.rcParams['legend.fancybox'] = True
        mpl.rcParams['font.size'] = 12.
        mpl.rcParams['xtick.labelsize'] = 12.
        mpl.rcParams['ytick.labelsize'] = 12.
        mpl.rcParams['figure.subplot.left'] = .15
        mpl.rcParams['figure.subplot.right'] = .95
        mpl.rcParams['figure.subplot.top'] = .99
        mpl.rcParams['figure.subplot.bottom'] = .17
        mpl.rcParams['figure.subplot.hspace'] = 0.05
        mpl.rcParams['figure.subplot.wspace'] = 0.05  
        fig, ax = plt.subplots(figsize=(8, 8),
                               subplot_kw={'projection': ccrs.PlateCarree()})

        # Plot the vegetation map
        img = ax.imshow(
            vegetation_crop,
            origin='upper',
            extent=[lon_min, lon_max, lat_min, lat_max],
            transform=ccrs.PlateCarree(),
            interpolation='nearest',
            cmap=cmap_veg,
            norm=norm_veg,
            zorder=0
        )

        #fire loc
        ax.scatter(lon,lat, s=100, facecolor='none', edgecolor='k', zorder=1, transform=ccrs.PlateCarree(), linewidth=2  )

        # Add map features
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.COASTLINE)
        #ax.add_feature(cfeature.LAND, edgecolor='black', alpha=0.2)
        #ax.add_feature(cfeature.OCEAN)
        #ax.add_feature(cfeature.LAKES, alpha=0.4)
        #ax.add_feature(cfeature.RIVERS)

        ax.set_title("Corine Land Cover", fontsize=20)
        cbar = plt.colorbar(img, ax=ax, orientation='horizontal',pad=.02)
        
        tick_locs = values_colorbar[:-1]+0.5*np.diff(values_colorbar) #legend_filtered.index.values + 0.5  # Center of bins
        
        cbar.set_ticks(tick_locs)
        # Set the tick labels using the 'label' column
        cbar.set_ticklabels(legend_filtered['label'].values, fontsize=20)
        # Optional: set label/title

        # Rotate tick labels by 45 degrees
        for label in cbar.ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right')  # Optional: improves alignment
    
        add_scalebar(ax, 2, location=(0.1, 0.05), linewidth=3, text_offset=0.01)

        # === sub fig cart location fire ===
        ax_loc = fig.add_axes([0.02, 0.72, 0.3, 0.3], projection=ccrs.PlateCarree())
        ax_loc.scatter(lon,lat, s=20, color='r',zorder=4)
        # Base map
        ax_loc.add_feature(cfeature.COASTLINE,zorder=3)
        ax_loc.add_feature(cfeature.BORDERS, linestyle=':',zorder=2)
        ax_loc.add_feature(cfeature.LAND, facecolor='lightgray',zorder=0)
        ax_loc.add_feature(cfeature.OCEAN, facecolor='lightblue',zorder=1)
        ax_loc.set_extent([-6, 10.5, 41, 51.5], crs=ccrs.PlateCarree())
        lat_dms = decimal_to_dms(lat)
        lon_dms = decimal_to_dms(lon)
        #XX_FIRE_LOC = "lon: {lon_dms[0]}°{lon_dms[1]}'{lon_dms[2]}\"\n"+f"lat: {lat_dms[0]}°{lat_dms[1]}'{lat_dms[2]}\""
        #ax_loc.set_title(f"lon: {lon_dms[0]}°{lon_dms[1]}'{lon_dms[2]}\"\n"+f"lat: {lat_dms[0]}°{lat_dms[1]}'{lat_dms[2]}\"" , fontsize=21) 
         
        fig.savefig(f"{params['general']['reportDir']}/{XX_FIRE_VEG_MAP}")
        plt.close(fig)

        #local pof image
        pof_j0 = ds_pof.sel(time=time_report+pd.Timedelta(days=0),method='nearest')['MODEL_FIRE']
        plot_pof(params,pof_j0, f'pof_{row["id_fire_event"]}_j0.png',extent=bounding_box_km(lon, lat),fireloc=(lon,lat),flag_scalebar=True) 
        pof_j1 = ds_pof.sel(time=time_report+pd.Timedelta(days=1),method='nearest')['MODEL_FIRE']
        plot_pof(params,pof_j1, f'pof_{row["id_fire_event"]}_j1.png',extent=bounding_box_km(lon, lat),fireloc=(lon,lat))
        pof_j2 = ds_pof.sel(time=time_report+pd.Timedelta(days=2),method='nearest')['MODEL_FIRE']
        plot_pof(params,pof_j2, f'pof_{row["id_fire_event"]}_j2.png',extent=bounding_box_km(lon, lat),fireloc=(lon,lat),flag_colorbar=True)
        XX_FIRE_ID=f"{row['id_fire_event']}"
        
        # === 1. Metadata ===
        XX_METADATA = f"Last obs - time: {time.strftime('%Y%m%dT%H:%MZ')} \quad FRP: {frp:5.2f} MW \quad Area: {area:5.2f} km$^{2}$ \quad lon: {lon_dms[0]}°{lon_dms[1]}'{lon_dms[2]} \quad lat: {lat_dms[0]}°{lat_dms[1]}'{lat_dms[2]}\""

        #dummy rgb and ir38
        shutil.copy( f"{params['general']['srcDir']}/fireReport/rgb_184_h0.png",f"{params['general']['reportDir']}/rgb_{XX_FIRE_ID}_h0.png")
        shutil.copy( f"{params['general']['srcDir']}/fireReport/rgb_184_h1.png",f"{params['general']['reportDir']}/rgb_{XX_FIRE_ID}_h1.png")
        shutil.copy( f"{params['general']['srcDir']}/fireReport/rgb_184_h2.png",f"{params['general']['reportDir']}/rgb_{XX_FIRE_ID}_h2.png")
        shutil.copy( f"{params['general']['srcDir']}/fireReport/ir38_184_h0.png",f"{params['general']['reportDir']}/ir38_{XX_FIRE_ID}_h0.png")
        shutil.copy( f"{params['general']['srcDir']}/fireReport/ir38_184_h1.png",f"{params['general']['reportDir']}/ir38_{XX_FIRE_ID}_h1.png")
        shutil.copy( f"{params['general']['srcDir']}/fireReport/ir38_184_h2.png",f"{params['general']['reportDir']}/ir38_{XX_FIRE_ID}_h2.png")


        #add content to line for text file
        linesFires.append(firePage(params,XX_FIRE_NAME, XX_FIRE_VEG_MAP, XX_FIRE_FRP,XX_METADATA,XX_FIRE_ID,XX_FFMNH_URL))


    generate_report(params,XX_TABLE_HERE, XX_NUMBER_OF_FIRE, time_report, linesFires)

    os.makedirs(f"{dir_report}/{time_report.strftime('%Y%m%d')}",exist_ok=True)
    shutil.copy(f"{params['general']['reportDir']}/fireReport.pdf", f"{dir_report}/{time_report.strftime('%Y%m%d')}/fires_FR_{time_last_geojson}.pdf")
    print(f"✅ PDF created: {dir_report}/fires_FR_{time_last_geojson}.pdf")
    
    #shutil.rmtree(params['general']['reportDir'])
