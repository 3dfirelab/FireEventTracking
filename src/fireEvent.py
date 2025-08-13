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

#########################
def getCloudMask(params, target_time, polygon):

    #find cloud mask file
    cloudmask_dir = Path(params['event']['dir_cloudMask'])

    # Collect all files with their datetime from filename
    files = []
    for root, dirs, filenames in os.walk(cloudmask_dir):
        for fn in filenames:
            if fn.endswith('.nc'):
                # Example: fci-cm-SILEX-2025218.0000.nc
                # Extract YYYYDOY.HHMM
                try:
                    doy_str = fn.split('-')[3]  # '2025218.0000.nc' part
                    yyyydoy, hm = doy_str.split('.')[:2]  # '2025218', '0000'
                    dt = pd.to_datetime(yyyydoy, format='%Y%j') \
                           + pd.Timedelta(hours=int(hm[:2]), minutes=int(hm[2:]))
                    files.append((dt.tz_localize('UTC'), Path(root) / fn))
                except Exception:
                    pass

    # Filter for dates <= target_time
    files_before = [(dt, f) for dt, f in files if dt <= target_time]

    if not files_before:
       return np.nan 

    # Pick the one closest to target_time
    closest_dt, closest_file = max(files_before, key=lambda x: x[0])

    ds = xr.open_dataset(closest_file)

    pdb.set_trace()


##########################
class Event: 
    _id_counter=0


    def __init__(self,params, cluster, ctr, crs, hs_all, gdf_postcode): 
        self.id_fire_event = Event._id_counter
        Event._id_counter += 1
        
        self.times = [cluster.end_time.tz_localize('UTC')]
        self.time_ranges = [(cluster.start_time.tz_localize('UTC'),cluster.end_time.tz_localize('UTC'))]
        self.cluster_fire_event = [ctr.cluster_fire_event]
        self.ctrs  = gpd.GeoDataFrame([{'geometry':ctr.geometry}] ,crs=crs, geometry='geometry')
        self.centers = [Point(cluster.x, cluster.y)]
        self.frps = [cluster.frp]

        self.fire_name = assign_single_fire_name(self, gdf_postcode)    

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
        
        self.areas = self.ctrs.geometry.area.to_list()

        #if  self.id_fire_event == 10:
        #    ax=plt.subplot(111)
        #    self.ctrs.iloc[-1:].plot(ax=ax,facecolor='none')
        #    self.hspots.iloc[-1:].plot(ax=ax)
        #    plt.show()

        self.id_fire_event_dad = []

        #self.cloudMask = getCloudMask(params, self.times[-1], self.ctrs.iloc[-1])


    def add(self, cluster, ctr, crs, hs_all):
        if self.testSimilarityOfEvent(cluster, ctr): return  
        
        self.times.append( cluster.end_time.tz_localize('UTC'))
        self.time_ranges.append((cluster.start_time.tz_localize('UTC'),cluster.end_time.tz_localize('UTC')))
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
      
        self.areas = self.ctrs.geometry.area.to_list()

        #if  self.id_fire_event == 10:
        #    ax=plt.subplot(111)
        #    self.ctrs.iloc[-1:].plot(ax=ax,facecolor='none')
        #    self.hspots.iloc[-1:].plot(ax=ax)
        #    plt.show()

    def add_ffmnhSimu(self, url):
        self.ffmnhUrl=url

    def testSimilarityOfEvent(self, cluster, ctr):
        flag_frp_same = False
        if cluster.frp != 0 : 
            if abs(self.frps[-1] - cluster.frp )/ cluster.frp < .05 : 
                flag_frp_same = True
        else: 
            if abs(self.frps[-1] - cluster.frp ) <= 0 : 
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

    def save(self, status, params, datetime_now=None, local_dir=None):
        if status == 'active':
            if local_dir != None:
                file_path = local_dir+'/Pickles_{:s}_{:s}/{:09d}.pkl'.format(status,datetime_now.strftime("%Y-%m-%d_%H%M"),self.id_fire_event)
            else:
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
    
        if len(glob.glob( params['event']['dir_data']+f'/Pickles_{"past"}/{self.id_fire_event:09d}_*' ) ) > 0 : 
            print(self.id_fire_event)
            print('file id is alreay there!!!!')
            pdb.set_trace()

        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

        if local_dir != None:
            return file_path,  params['event']['dir_data']+'/Pickles_{:s}_{:s}/{:09d}.pkl'.format(status,datetime_now.strftime("%Y-%m-%d_%H%M"),self.id_fire_event)

##########################
def load_fireEvent(file_path):
    """
    Load an Event instance from disk using pickle.

    Parameters:
    - file_path (str): Path to the `.pkl` file containing the saved Event instance.

    Returns:
    - Event: The deserialized Event instance.
    """
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except: 
        pdb.set_trace()

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


###########################
def assign_single_fire_name(self, gdf_postcode: gpd.GeoDataFrame) -> str:
    """
    Assign a fire_name to the last fire event using postcode spatial information.

    Parameters
    ----------
    gdf_postcode : GeoDataFrame
        Postcode boundaries with columns: 'geometry', 'NSI_CODE', 'COMM_NAME', 'CNTR_CODE', 'NUTS_CODE'.

    Returns
    -------
    str
        Generated fire_name string.
    """
    # Get last center and time
    point = self.centers[-1]
    time = pd.to_datetime(self.times[-1])

    # Create temporary GeoDataFrame
    gdf_point = gpd.GeoDataFrame(
        [{'geometry': point, 'time': time}],
        geometry='geometry',
        crs=gdf_postcode.crs
    )

    # Spatial join
    joined = gpd.sjoin(
        gdf_point,
        gdf_postcode[["geometry", "NSI_CODE", "COMM_NAME", "CNTR_CODE", "NUTS_CODE"]],
        how="left",
        predicate="intersects"
    )

    row = joined.iloc[0]  # Only one point

    # Compose fire_name
    fire_name = (
        "fire_" +
        (row["CNTR_CODE"] if pd.notnull(row["CNTR_CODE"]) else "XX") + "_" +
        (str(row["NSI_CODE"]) if pd.notnull(row["NSI_CODE"]) else "00000") + "_" +
        (row["COMM_NAME"].replace(" ", "") if pd.notnull(row["COMM_NAME"]) else "UnknownCity") + "_" +
        (str(row["NUTS_CODE"]) if pd.notnull(row["NUTS_CODE"]) else "00000") + "_" +
        time.strftime("%Y%m%d")
    )

    return fire_name


