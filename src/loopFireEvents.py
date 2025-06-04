import numpy as np 
import pandas as pd
import os 
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt 
import sys
from pathlib import Path
import importlib 
from datetime import datetime, timezone, timedelta
from sklearn.cluster import DBSCAN
from shapely.geometry import Point, MultiPoint
import geopandas as gpd
import pdb 
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
import rioxarray
import rasterio
from rasterio.features import rasterize
import xarray as xr
warnings.filterwarnings("error", category=pd.errors.SettingWithCopyWarning)
import requests 
import pyforefire as forefire

#home brewed
import fireEvent as fireEventTool
import fireEventTracking
import interpArrivalTime

##########################
def loopFireEvents4ROS(params, currentTime):

    fireEvents = fireEventTool.load_fireEvents(params,currentTime)
    
    WGS84proj  = pyproj.Proj('EPSG:4326')
    UTMproj  = pyproj.Proj(params['general']['crs'])
    lonlat2xy = pyproj.Transformer.from_proj(WGS84proj,UTMproj)
    xy2lonlat = pyproj.Transformer.from_proj(UTMproj,WGS84proj)
    
    out_dir_srtm = params['event']['dir_data']+'/SRTM/'
    os.makedirs(out_dir_srtm, exist_ok=True)
    
    for ii, fireEvent in enumerate(fireEvents):

        #here we set the same approach as for wildfiresat
        #compute arrival time map for < 6h interval and compute ros on these subsection -- this get_bamp un ff2bmap on andromeda
        #this will output raster of ROS with big gaps where we are missing data 
        
        if fireEvent is None: continue
   
        #if len(fireEvent.times)> 3:
        if fireEvent.id_fire_event == 313 :
    
            out_dir_event = '{:s}/Pickles_{:s}_{:s}/{:09d}/'.format(params['event']['dir_data'],'active',
                                                             currentTime.strftime("%Y-%m-%d_%H%M"),fireEvent.id_fire_event)
            os.makedirs(out_dir_event, exist_ok=True)
            firename = '{:d}-{:s}'.format(fireEvent.id_fire_event,fireEvent.times[-1].strftime('%Y%m%d.%H%M'))
     
           
            #set raster grid and terrain
            #----
            fireEvent.ctrs.to_crs(params['general']['crs'],inplace=True)
            xmin,ymin,xmax,ymax = fireEvent.ctrs.total_bounds
            buffer_dist = 500
            xmin -= buffer_dist
            ymin -= buffer_dist
            xmax += buffer_dist
            ymax += buffer_dist
            Lx = xmax-xmin 
            Ly = ymax-ymin 
            dx = 50
            buffer = 1
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
    
            grid_lat, grid_lon = xy2lonlat.transform(maps_fire.grid_e.flatten()+ dx/2, maps_fire.grid_n.flatten()+ dx/2)
            xx_lon = grid_lon.reshape(maps_fire.shape)[:,0]
            yy_lat = grid_lat.reshape(maps_fire.shape)[0,:]
    
            terrainFile = "{:s}/srtm_data_{:s}.tif".format(out_dir_srtm,firename)
            try: 
                if not(os.path.isfile(terrainFile)):
                    buffer_ll = 0.04
                    bounds = (float(grid_lon.min())-buffer_ll, float(grid_lat.min())-buffer_ll, float(grid_lon.max())+buffer_ll, float(grid_lat.max()+buffer_ll))  # Example coordinates
                    elevation.clip(bounds=bounds, output=terrainFile, margin='0')
                    elevation.clean()
            except: 
                pdb.set_trace()
            terrain_srtm = rioxarray.open_rasterio(terrainFile)
            terrain_srtm = terrain_srtm.rio.reproject(params['general']['crs'])
            terrain_srtm = terrain_srtm.interp(x=x, y=y)
            maps_fire.terrain = terrain_srtm.isel(band=0).T
           

            #fill up the raw arrival time map.
            #----
            arrT = terrain_srtm.isel(band=0).copy()
            arrT.values[:,:] = -9999
            diffTprev = arrT.copy()
            ds = xr.Dataset({
                'arrT': arrT,
                'diffTprev': diffTprev
                })

            df = fireEvent.ctrs
            df['time'] =  [(xx-fireEvent.times[0]).total_seconds() for xx in fireEvent.times ]
           
            res_x = float(ds.x[1] - ds.x[0])
            res_y = float(ds.y[0] - ds.y[1])  # Should be negative due to decreasing y
            transform = rasterio.transform.from_origin(float(ds.x[0]), float(ds.y[0]), res_x, res_y)
            out_shape = (ds.sizes['y'], ds.sizes['x'])

            for ii, row in df.iterrows():
                # Create (geometry, value) pairs
                shapes = [(row.geometry, row.time)]

                # Rasterize
                rasterized = rasterize(
                    shapes=shapes,
                    out_shape=out_shape,
                    transform=transform,
                    fill=ds.attrs.get('_FillValue', -32768),
                    dtype='float32',
                    all_touched=True)
                idx = np.where( (rasterized>0) & (ds.arrT<0) )
                ds.arrT.values[idx] = row.time
                diffT = row.time - df.time[ii-1] if ii>0 else -9999
                ds.diffTprev.values[idx] = diffT

            #ax = plt.subplot(111)
            #ds.diffTprev.where(ds.arrT>=0).plot(ax=ax,cmap='viridis')
            #fireEvent.ctrs.plot(facecolor='none',ax=ax)
            #plt.show()

            #compute ros
            #----
            maps_fire.plotMask = np.where(ds.arrT.T.values>0,2,0)


            #interpolate arrivat time
            arrivalTime_edgepixel = np.zeros_like(maps_fire.grid_e)
            arrivalTime_ignition_seed = None
            params_interp = {}
            params_interp['plot_name'] =  firename                      
            params_interp['flag_parallel'] = True            
            params_interp['ros_max'] = 10. 
            params_interp['flag_use_time_reso_constraint'] = False
            params_interp['interpolate_betwee_ff'] = 'fsrbf'
            params_interp['distance_interaction_rbf'] = 50.
            arrivalTime_interp, arrivalTime_FS, arrivalTime_clean = \
                                interpArrivalTime.interpolate_arrivalTime(              \
                                out_dir_event, maps_fire, pyproj.CRS.from_epsg(params['general']['crs']),           \
                                ds.arrT.T.values, arrivalTime_edgepixel,                     \
                                params_interp,                                                 \
                                #flag_interpolate_betwee_ff= 'rbf',                     \
                                flag_plot                 = True,                      \
                                frame_time_period         = 0,                          \
                                set_rbf_subset            = None,                       \
                                arrivalTime_ignition_seed =  arrivalTime_ignition_seed, \
                                                                                        )
            #compute ROS
            ##########   
            normal_x, normal_y, ros, ros_qc = ROSlib.compute_ros( maps_fire, arrivalTime_interp, arrivalTime_clean, ros_max=params_interp['ros_max']) 
            
            
#############################
def loopFireEvents4FireGrowth(params, currentTime):
    fireEvents = fireEventTool.load_fireEvents(params,currentTime)
    
    WGS84proj  = pyproj.Proj('EPSG:4326')
    UTMproj  = pyproj.Proj(params['general']['crs'])
    lonlat2xy = pyproj.Transformer.from_proj(WGS84proj,UTMproj)
    xy2lonlat = pyproj.Transformer.from_proj(UTMproj,WGS84proj)
    
    #out_dir_srtm = params['event']['dir_data']+'/FORFIRE/'
    #os.makedirs(out_dir_srtm, exist_ok=True)
    
    for ii, fireEvent in enumerate(fireEvents):

        if fireEvent is None: continue
   
        out_dir_event = '{:s}/Pickles_{:s}_{:s}/FOREFIRE/{:09d}/'.format(params['event']['dir_data'],'active',
                                                         currentTime.strftime("%Y-%m-%d_%H%M"),fireEvent.id_fire_event)
        os.makedirs(out_dir_event, exist_ok=True)
        firename = '{:d}-{:s}'.format(fireEvent.id_fire_event,fireEvent.times[-1].strftime('%Y%m%d.%H%M'))
    
        igni = {}
        igni['time'] = fireEvent.times[-1]
        igni['perimeter'] = fireEvent.ctrs.to_crs(4326).iloc[-1]
        

        #collect input data.
        center = igni['perimeter'].geometry.centroid
        lat_c_025 = np.round(center.y*4)/4
        lon_c_025 = np.round(center.x*4)/4
        local_tile = f"{dirDataTileFOREFIRE}/data_{lat_c_025}_{lon_c_025}.nc"

        if ~os.path.isfile(local_tile):
            url = f"https://forefire.univ-corse.fr/saphir/FF/tiles/{lat_c_025:g}_{lon_c_025:g}/data.nc"
            response = requests.get(url)
            with open(local_tile, "wb") as f:
                f.write(response.content)
        
        ds = xr.open_dataset(local_tile)
       
        #move file to out_dir_event
        shutil.move(local_tile, out_dir_event)

        ff = forefire.ForeFire()

        myCmd = "FireDomain[sw=(0.,0.,0.);ne=(%f,%f,0.);t=0.]" % (sizeX, sizeY)

        # Execute the command
        ff.execute(myCmd)
        
        #get coords for the data.

        '''
        #if needed, code below to load the 9 tiles
        #get the name of the 9 tiles around
        offsets = [(-0.25, 0.25),( 0.00, 0.25),( 0.25, 0.25), 
                  (-0.25, 0.00),( 0.00, 0.00),( 0.25, 0.00),
                  (-0.25,-0.25),( 0.00,-0.25),( 0.25,-0.25)]
        local_tile_all = []
        for dlon, dlat in offsets:
            lat_c_025_ = lat_c_025 + dlat 
            lon_c_025_ = lon_c_025 + dlon
            
            local_tile = f"{dirDataTileFOREFIRE}/data_{lat_c_025_}_{lon_c_025_}.nc"
            print(local_tile)
            if not(os.path.isfile(local_tile)):
                url = f"https://forefire.univ-corse.fr/saphir/FF/tiles/{lat_c_025:g}_{lon_c_025:g}/data.nc"
                response = requests.get(url)
                with open(local_tile, "wb") as f:
                    f.write(response.content)
            local_tile_all.append(local_tile)
        '''
        


        pdb.set_trace()

           


#############################
if __name__ == '__main__':
#############################
    importlib.reload(fireEventTool)
    
    dirDataTileFOREFIRE = '/mnt/dataEstrella2/FOREFIRE_TILE'

    #init
    params = fireEventTracking.init('SILEX') 
    currentTime = datetime.strptime('2025-05-30_1300', '%Y-%m-%d_%H%M')
   
    #loop for FOREFIRE
    loopFireEvents4FireGrowth(params, currentTime )

    #loop for ROS
    #loopFireEvents4ROS(params, currentTime )



