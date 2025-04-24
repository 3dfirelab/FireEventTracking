import numpy as np 
import pandas as pd
import os 
import matplotlib as mpl
import matplotlib.pyplot as plt 
import sys
from pathlib import Path
import importlib 
from datetime import datetime, timezone, timedelta
from sklearn.cluster import DBSCAN
from shapely.geometry import Point, MultiPoint
import geopandas as gpd
import pdb 
import alphashape
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
from scipy.spatial.distance import cdist
import warnings
import pickle
import shutil
import glob
import pyproj 
import elevation
warnings.filterwarnings("error", category=pd.errors.SettingWithCopyWarning)

#home brewed
import fireEvent
import fireEventTracking

############################
def ros():
   return None


############################
def load_fireEvents(params,currentTime): 
    fireEvents_files = sorted(glob.glob( params['event']['dir_data']+'/Pickles_{:s}_{:s}/*.pkl'.format('active',currentTime.strftime("%Y-%m-%d_%H%M"))))
    
    fireEvents = []
    ii = 0
    for id_, event_file in enumerate(fireEvents_files):
        event = fireEvent.load_fireEvent(event_file)
        while ii < event.id_fire_event:
            fireEvents.append(None)
            ii += 1
        fireEvents.append( fireEvent.load_fireEvent(event_file) ) 
        ii += 1
    fireEvent.Event._id_counter = fireEvents[-1].id_fire_event +1

    return fireEvents

#############################
if __name__ == '__main__':
#############################

    params = fireEventTracking.init('AVEIRO') 

    currentTime = datetime.strptime('2024-09-17_2000', '%Y-%m-%d_%H%M')
    fireEvents = load_fireEvents(params,currentTime)
    
    WGS84proj  = pyproj.Proj('EPSG:4326')
    UTMproj  = pyproj.Proj(params['general']['crs'])
    lonlat2xy = pyproj.Transformer.from_proj(WGS84proj,UTMproj)
    xy2lonlat = pyproj.Transformer.from_proj(UTMproj,WGS84proj)
    
    out_dir_srtm = params['event']['dir_data']+'srtm/'
    os.makedirs(out_dir_srtm, exist_ok=True)
    
    for ii, fireEvent in enumerate(fireEvents):

        #here we set the same approach as for wildfiresat
        #compute arrival time map for < 6h interval and compute ros on these subsection -- this get_bamp un ff2bmap on andromeda
        #this will output raster of ROS with big gaps where we are missing data 
        
        if fireEvent is None: continue
   
        if len(fireEvent.times)> 3:
           
            #set raster grid
            xmin,ymin,xmax,ymax = fireEvent.ctrs.total_bounds
            Lx = xmax-xmin
            Ly = ymax-ymin
            dx = 100
            buffer = 4
            nx = int(Lx//dx + buffer)
            ny = int(Ly//dx + buffer)

            arrivalTime = np.zeros([nx,ny])
            x = np.arange(xmin - buffer/2*dx, xmax - buffer/2*dx, dx)
            y = np.arange(ymin - buffer/2*dx, ymax - buffer/2*dx, dx)
            grid_n, grid_e = np.meshgrid(y,x)

            maps_fire = np.zeros(grid_e.shape,dtype=([('grid_e',float),('grid_n',float),('plotMask',float),('terrain',float)]))
            maps_fire = maps_fire.view(np.recarray)
            maps_fire.grid_e = grid_e
            maps_fire.grid_n = grid_n
    
            #lon lat at the pixel center
            grid_lat, grid_lon = xy2lonlat.transform(maps_fire.grid_e.flatten()+ dx/2, maps_fire.grid_n.flatten()+ dx/2)
            xx_lon = grid_lon.reshape(maps_fire.shape)[:,0]
            yy_lat = grid_lat.reshape(maps_fire.shape)[0,:]
    
            firename = '{:d}-{:s}'.format(fireEvent.id_fire_event,fireEvent.times[-1].strftime('%Y%m%d.%H%M'))
            terrainFile = "{:s}/srtm_data_{:s}.tif".format(out_dir_srtm,firename)
            if not(os.path.isfile(terrainFile)):
                bounds = (float(grid_lon.min()), float(grid_lat.min()) , float(grid_lon.max()), float(grid_lat.max()))  # Example coordinates
                elevation.clip(bounds=bounds, output=terrainFile)
                elevation.clean()
            terrain_srtm = rioxarray.open_rasterio(terrainFile)
            terrain_srtm = terrain_srtm.rio.reproject(params['projection'])
            terrain_srtm = terrain_srtm.interp(x=xx, y=yy)
            maps_fire.terrain = terrain_srtm.isel(band=0).T


            sys.exit()

