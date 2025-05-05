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


#homebrewed
import interpArrivalTime
import ros as ROSlib

importlib.reload(interpArrivalTime)
importlib.reload(ROSlib)

##########################
class Event: 
    _id_counter=0


    def __init__(self, cluster, ctr, crs, hs_all): 
        self.id_fire_event = Event._id_counter
        Event._id_counter += 1
        
        self.times = [cluster.end_time]
        self.time_ranges = [(cluster.start_time,cluster.end_time)]
        self.cluster_fire_event = [ctr.cluster_fire_event]
        self.ctrs  = gpd.GeoDataFrame([{'geometry':ctr.geometry}] ,crs=crs, geometry='geometry')
        self.centers = [Point(cluster.x, cluster.y)]
        self.frps = [cluster.frp]

        #keep only the one that match self.times
        hsidx_ = cluster.indices_hs
        hs_ = hs_all.loc[hsidx_]
        hs_ = hs_[hs_['timestamp']==self.times[-1]]
        self.indices_hs = [hs_.index]

        points = hs_all.loc[self.indices_hs[-1]].geometry.tolist()
        multi_point_geom = MultiPoint(points)
        self.hspots = gpd.GeoDataFrame([{'geometry':multi_point_geom}] ,crs=crs, geometry='geometry')
        hsfrps =  hs_all.loc[self.indices_hs[-1]].frp.tolist()
        self.hspots['frp'] = None 
        self.hspots.at[0,'frp'] = hsfrps
       
        #if  self.id_fire_event == 10:
        #    ax=plt.subplot(111)
        #    self.ctrs.iloc[-1:].plot(ax=ax,facecolor='none')
        #    self.hspots.iloc[-1:].plot(ax=ax)
        #    plt.show()

        self.id_fire_event_dad = []


    def add(self, cluster, ctr, crs, hs_all):
        if self.testSimilarityOfEvent(cluster, ctr): return  
        
        self.times.append( cluster.end_time)
        self.time_ranges.append((cluster.start_time,cluster.end_time))
        self.cluster_fire_event.append(ctr.cluster_fire_event)
        self.ctrs = pd.concat([self.ctrs, gpd.GeoDataFrame([{'geometry':ctr.geometry}],crs=crs, geometry='geometry')]).reset_index(drop=True)
        self.centers.append(Point(cluster.x, cluster.y))
        self.frps.append(cluster.frp)

        #keep only the one that match self.times
        hsidx_ = cluster.indices_hs
        hs_ = hs_all.loc[hsidx_]
        hs_ = hs_[hs_['timestamp']==self.times[-1]]
        self.indices_hs.append( hs_.index )
        
        points = hs_all.loc[self.indices_hs[-1]].geometry.tolist()
        multi_point_geom = MultiPoint(points)
        hspots_ = gpd.GeoDataFrame([{'geometry':multi_point_geom}] ,crs=crs, geometry='geometry')
        self.hspots =  pd.concat([self.hspots, hspots_]).reset_index(drop=True)
        
        hsfrps =  hs_all.loc[self.indices_hs[-1]].frp.tolist()
        self.hspots.at[self.hspots.index[-1], 'frp'] = hsfrps
       
        #if  self.id_fire_event == 10:
        #    ax=plt.subplot(111)
        #    self.ctrs.iloc[-1:].plot(ax=ax,facecolor='none')
        #    self.hspots.iloc[-1:].plot(ax=ax)
        #    plt.show()

    def testSimilarityOfEvent(self, cluster, ctr):
        flag_frp_same = False
        if abs(self.frps[-1] - cluster.frp )/ cluster.frp < .05 : 
            flag_frp_same = True
        flag_geo_same= False
        if (self.ctrs.iloc[-1].geometry.geom_type == 'Polygon') and (ctr.geometry.geom_type == 'Polygon'): 
            if average_min_distance(self.ctrs.iloc[-1].geometry, ctr.geometry, symmetric=False) < 20 :
                flag_geo_same = True
        else: 
            flag_geo_same = True

        if flag_geo_same and flag_frp_same: 
            return True
        else: 
            return False

    def merge(self, event): 
        [self.times.append( time_) for time_ in event.times]
        [self.time_ranges.append(time_range_) for time_range_ in event.time_ranges]
        [self.cluster_fire_event.append(cluster_fire_event_) for cluster_fire_event_ in event.cluster_fire_event ]
        [self.centers.append( point_) for point_ in event.centers]
        [self.frps.append(frp_) for frp_ in event.frps]
        [self.indices_hs.append(indices_hs_) for indices_hs_ in event.indices_hs]
        self.ctrs = pd.concat([self.ctrs, event.ctrs]).reset_index(drop=True)
        self.hspots = pd.concat([self.hspots, event.hspots]).reset_index(drop=True)
        
        #reorder in time sequence
        sorted_indices = np.argsort(self.times)
        
        self.times = [self.times[ii] for ii in sorted_indices ]
        self.time_ranges = [self.time_ranges[ii] for ii in sorted_indices ]
        self.cluster_fire_event = [self.cluster_fire_event[ii] for ii in sorted_indices ]
        self.centers = [self.centers[ii] for ii in sorted_indices ]
        self.frps = [self.frps[ii] for ii in sorted_indices ]
        self.indices_hs = [self.indices_hs[ii] for ii in sorted_indices ]
        self.ctrs = self.ctrs.iloc[sorted_indices].reset_index(drop=True)
        self.hspots = self.hspots.iloc[sorted_indices].reset_index(drop=True)

    def mergeWith(self, idx_dad):
        for i in idx_dad:
            self.id_fire_event_dad.append(i)

    def save(self, status, params, datetime_now=None):
        if status == 'active':
            file_path = params['event']['dir_data']+'/Pickles_{:s}_{:s}/{:09d}.pkl'.format(status,datetime_now.strftime("%Y-%m-%d_%H%M"),self.id_fire_event)
        elif status == 'past':
            file_path = params['event']['dir_data']+'/Pickles_{:s}/{:09d}_{:s}.pkl'.format(status,self.id_fire_event,self.times[-1].strftime("%Y-%m-%d_%H%M"))

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if status == 'past': 
            if os.path.isfile(file_path):
                return 
        """
        Save an Event instance to disk using pickle.

        Parameters:
        - event_instance (Event): The instance to be saved.
        - file_path (str): Path where the file will be stored. Should end in `.pkl`.
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)


##########################
def load_fireEvent(file_path):
    """
    Load an Event instance from disk using pickle.

    Parameters:
    - file_path (str): Path to the `.pkl` file containing the saved Event instance.

    Returns:
    - Event: The deserialized Event instance.
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)


############################
def load_fireEvents(params,currentTime): 
    fireEvents_files = sorted(glob.glob( params['event']['dir_data']+'/Pickles_{:s}_{:s}/*.pkl'.format('active',currentTime.strftime("%Y-%m-%d_%H%M"))))
    
    fireEvents = []
    ii = 0
    for id_, event_file in enumerate(fireEvents_files):
        event = load_fireEvent(event_file)
        while ii < event.id_fire_event:
            fireEvents.append(None)
            ii += 1
        fireEvents.append( load_fireEvent(event_file) ) 
        ii += 1
    Event._id_counter = fireEvents[-1].id_fire_event +1

    return fireEvents

##########################
def average_min_distance(poly1, poly2, symmetric=True):
    # Extract coordinates
    coords1 = np.array(poly1.exterior.coords)
    coords2 = np.array(poly2.exterior.coords)
    
    # Compute pairwise distances
    distances = cdist(coords1, coords2)
    
    # Average of min distances from poly1 to poly2
    d1 = np.mean(np.min(distances, axis=1))
    
    if symmetric:
        d2 = np.median(np.min(distances, axis=0))
        return 0.5 * (d1 + d2)
    else:
        return d1 


##########################
def loopFireEvents(params, currentTime):

    fireEvents = load_fireEvents(params,currentTime)
    
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
            




