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


############################
if __name__ == '__main__':
############################ 
    inputName ='SILEX-MF'
    sensorName = 'FCI'
    log_dir = os.path.dirname(os.path.abspath(__file__)) +'/../log_doc/'

    params = FET.init(inputName,sensorName,log_dir)
    params['general']['srcDir']= os.path.dirname(__file__)
   
    outDir = params['general']['root_data'] + 'FOREFIRE/'
    os.makedirs(outDir, exist_ok=True)

    geojsons = sorted(glob.glob(params['event']['dir_geoJson']+'/*.geojson'))
    last_geojson = geojsons[-1]
    time_last_geojson = os.path.basename(last_geojson).split('.geojs')[0].split('ents-')[1]
    time_report = pd.to_datetime(time_last_geojson, format='%Y-%m-%d_%H%M')


    # Load the GeoJSON file
    gdf = gpd.read_file(last_geojson)

    # Filter French fires by 'name' field
    gdf_france = gdf[gdf["name"].str.contains("_FR_", na=False)].copy()

    transformer = Transformer.from_crs("EPSG:{:d}".format(params['general']['crs']), "EPSG:4326", always_xy=True)

    #
    # loop over all french fire in last geojson
    #
    for _, row in gdf_france.iterrows():
        name = row["name"]
        frp = row["frp"]
        time = row["time"]
        image_url = row["image"]  # This is the FRP time series image
        ffmnh_url = row["ffmnhUrl"]    

        #pt = wkt_loads(row.center)
        #x, y = pt.x, pt.y
        #lon, lat = transformer.transform(x,y)
       
        output_file = f"{outDir}/ffmnh-{row['id_fire_event']}.nc"
            
        if os.path.isfile(output_file): continue
        if 'http' not in  ffmnh_url: continue
        print(name)

        response = requests.get(f"{ffmnh_url}/compiled_data.nc")
        if response.status_code == 200:
            with open(output_file, "wb") as f:
                f.write(response.content)
            #print(f"Downloaded to {output_file}")
        #else:
            #print(f"Failed to download. Status code: {response.status_code}")

