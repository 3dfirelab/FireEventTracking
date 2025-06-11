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
import socket
import argparse
import re 
from bisect import bisect_left
import rasterio
import xarray as xr 
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation

warnings.filterwarnings("error", category=pd.errors.SettingWithCopyWarning)

#home brewed
import hstools
import fireEvent
import os
import subprocess


#########################################
def mount_sftp_with_sshfs(mount_point):
    import subprocess
    import os

    host = os.getenv("SFTP_HOST")
    user = os.getenv("SFTP_USER")
    password = os.getenv("SFTP_PASS")
    port = os.getenv("SFTP_PORT", "22")
    remote_path = os.getenv("SFTP_REMOTE", "~")

    if not all([host, user, password, mount_point]):
        raise ValueError("SFTP_HOST, SFTP_USER, SFTP_PASS, and mount_point must be set")

    os.makedirs(mount_point, exist_ok=True)

    sshfs_cmd = [
        "sshpass", "-p", password,
        "sshfs",
        f"{user}@{host}:",
        mount_point,
        "-p", port,
        "-o", "reconnect",
        "-o", "StrictHostKeyChecking=no"
    ]

    print("Mounting:", " ".join(sshfs_cmd))

    try:
        subprocess.run(sshfs_cmd, check=True)
        print(f"Mounted {user}@{host}:{remote_path} -> {mount_point}")
    except subprocess.CalledProcessError as e:
        print("Mount failed:", e)
        sys.exit()

#############################
def create_gdf_fireEvents(fireEvents):
    
    filtered_fireEvents = [x for x in fireEvents if x is not None]
    
    gdf_activeEvent = pd.concat( [fireEvent.ctrs.iloc[-1:] for fireEvent in  filtered_fireEvents] ).reset_index().drop(columns=['index'])

    gdf_activeEvent['time'] = [fireEvent.times[-1] for fireEvent in  filtered_fireEvents]
    gdf_activeEvent['center'] = [fireEvent.centers[-1] for fireEvent in  filtered_fireEvents] 
    gdf_activeEvent['frp'] = [fireEvent.frps[-1] for fireEvent in  filtered_fireEvents] 
    gdf_activeEvent['id_fire_event'] = [fireEvent.id_fire_event for fireEvent in  filtered_fireEvents]
    gdf_activeEvent = gdf_activeEvent.set_index('id_fire_event')

    return gdf_activeEvent


####################################################
def init(config_name, log_dir):
    script_dir = Path(__file__).resolve().parent
    params = hstools.load_config(str(script_dir)+f'/../config/config-{config_name}.yaml')
  
    #if params['general']['use_sedoo_drive']:
    #    #mount SEDOO drive.
    #    mount_sftp_with_sshfs(params['general']['sedoo_mountpath'])
    #pdb.set_trace()
    
    if socket.gethostname() == 'moritz': 
        params['hs']['dir_data'] = params['hs']['dir_data'].replace('/mnt/data3/','/mnt/dataEstrella2/')
        params['event']['dir_data'] = params['event']['dir_data'].replace('/mnt/data3/','/mnt/dataEstrella2/')
    
    if socket.gethostname() == 'pc70852': 
        params['hs']['dir_data'] = params['hs']['dir_data'].replace('/mnt/data3/','/home/paugam/')
        params['event']['dir_data'] = params['event']['dir_data'].replace('/mnt/data3/','/home/paugam/')

    #create dir
    os.makedirs(params['event']['dir_data'],exist_ok=True)
    os.makedirs(log_dir,exist_ok=True)
    if 'dir_geoJson' in params['event'].keys():
        os.makedirs(params['event']['dir_geoJson'],exist_ok=True)
    

    return params
  

##################################################################################
def filter_points_by_mask(gdf: gpd.GeoDataFrame, mask_da: xr.DataArray, mask_value=1) -> gpd.GeoDataFrame:
    """
    Remove points from a GeoDataFrame where the corresponding value in a mask DataArray equals `mask_value`.

    Parameters:
    -----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame with Point geometries.
    mask_da : xarray.DataArray
        2D spatial mask with coordinates x (eastings) and y (northings).
    mask_value : int or float, optional
        The value in the mask used for exclusion (default is 1).

    Returns:
    --------
    geopandas.GeoDataFrame
        Filtered GeoDataFrame with only points where mask != mask_value.
    """

    # Ensure CRS matches
    crs_gdf = gdf.crs
    if gdf.crs != mask_da.rio.crs:
        gdf = gdf.to_crs(mask_da.rio.crs)

    # Extract x and y coordinates from the GeoDataFrame
    x_coords = gdf.geometry.x.values
    y_coords = gdf.geometry.y.values

    # Sample the mask DataArray at point locations using nearest neighbor
    sampled_vals = mask_da.sel(
        x=xr.DataArray(x_coords, dims="points"),
        y=xr.DataArray(y_coords, dims="points"),
        method="nearest"
    ).values

    # Filter points where mask value is not equal to the exclusion value
    keep_mask = sampled_vals != mask_value
    return gdf[keep_mask].reset_index(drop=True).to_crs(crs_gdf)


####################################################
#def perimeter_tracking(params, start_datetime,end_datetime, flag_restart=False):
def perimeter_tracking(params, start_datetime, flag_restart=False):
    
    start_datetime = datetime.strptime(f'{start_datetime}', '%Y-%m-%d_%H%M' ).replace(tzinfo=timezone.utc)
    #end_datetime   = datetime.strptime(f'{end_datetime}', '%Y-%m-%d_%H%M').replace(tzinfo=timezone.utc)

    #print('Load HS density')
    if not(os.path.isfile(f'{src_dir}/../data_local/mask_hs_600m_europe.nc')): 
        print('generate mask for fix hs ...')
        print('using VIIRS HS density from 2024 and polygon mask from OSM')
        with rasterio.open(params['event']['file_HSDensity_ESAWorldCover']) as src:
            HSDensity = src.read(1, masked=True)  # Use masked=True to handle nodata efficiently
            transform = src.transform
            crs = src.crs
            threshold = params['event']['threshold_HSDensity_ESAWorldCover']

        # Apply mask directly using NumPy vectorization
        mask_HS = (HSDensity > threshold).astype(np.uint8)

        # Build coordinate arrays using affine transform (faster with linspace)
        height, width = mask_HS.shape
        x0, dx = transform.c, transform.a
        y0, dy = transform.f, transform.e

        x_coords = x0 + dx * np.arange(width)
        y_coords = y0 + dy * np.arange(height)

        # Create DataArray and attach CRS
        maskHS_da = xr.DataArray(
            mask_HS,
            dims=["y", "x"],
            coords={"y": y_coords, "x": x_coords},
        ).rio.write_crs(crs, inplace=False)
        
        
        #add OSM industrial polygon to the mask
        indusAll = gpd.read_file(params['event']['file_polygonIndus_OSM']).to_crs(crs) 
       
        #transform = rasterio.transform.from_bounds(
        #    west=maskHS_da.lon.min().item(),
        #    south=maskHS_da.lat.min().item(),
        #    east=maskHS_da.lon.max().item(),
        #    north=maskHS_da.lat.max().item(),
        #    width=maskHS_da.sizes['x'],
        #    height=maskHS_da.sizes['y']
        #)

        out_shape = (maskHS_da.sizes['y'], maskHS_da.sizes['x'])

        # 3. Rasterize: burn value 1 wherever a polygon touches a pixel
        mask_array = rasterize(
            [(geom, 1) for geom in indusAll.geometry],
            out_shape=out_shape,
            transform=transform,
            fill=0,
            all_touched=True,  # key to ensure touching pixels are included
            dtype='uint8'
        )

        # 4. Create a new DataArray aligned with maskHS_da
        mask_rasterized = xr.DataArray(
            mask_array,
            dims=("y", "x"),
            coords={"y": maskHS_da.y, "x": maskHS_da.x}
        )

        # 5. Update maskHS_da where the rasterized mask is 1
        maskHS_da = xr.where(mask_rasterized == 1, 1, maskHS_da)
        
        #apply a dilatation
        footprint = np.ones((3, 3), dtype=bool)
        dilated_mask = binary_dilation(maskHS_da.values, structure=footprint)
        maskHS_da = xr.DataArray(
            dilated_mask,
            dims=("y", "x"),
            coords={"y": maskHS_da.y, "x": maskHS_da.x}
        )

        maskHS_da.to_netcdf(f'{src_dir}/../data_local/mask_hs_600m_europe.nc')

    else: 
        maskHS_da = xr.open_dataarray(f'{src_dir}/../data_local/mask_hs_600m_europe.nc').rio.write_crs("EPSG:4326", inplace=False)

    '''
    #load HS mask from ESAWorldCover
    print('load HS density')
    with rasterio.open(params['event']['file_HSDensity_ESAWorldCover']) as src:
        # Print basic metadata
        #print("CRS:", src.crs)
        #print("Bounds:", src.bounds)
        #print("Width, Height:", src.width, src.height)
        #print("Count (bands):", src.count)
        HSDensity = src.read(1)
        HSDensity_transform = src.transform
    mask_HS = np.where(HSDensity>params['event']['threshold_HSDensity_ESAWorldCover'], 1, 0)
    # Get shape
    height, width = HSDensity.shape

    # Compute coordinates from affine transform
    x_coords = np.arange(width) * HSDensity_transform.a + HSDensity_transform.c
    y_coords = np.arange(height) * HSDensity_transform.e + HSDensity_transform.f

    # Note: affine.e is usually negative (from top-left), so y decreases downwards

    # Create DataArray
    maskHS_da = xr.DataArray(
                                mask_HS,
                                dims=["y", "x"],
                                coords={"y": y_coords, "x": x_coords},
                                name="HSDensity"
                            )
    maskHS_da.rio.write_crs("EPSG:4326", inplace=True)
    '''
    #load fire event
    fireEvents = []
    pastFireEvents = []
    flag_PastFireEvent = False
    hsgdf_all_raw = None
 
    #select last data
    last_hs_saved = sorted(glob.glob("{:s}/hotspots-*.gpkg".format(params['event']['dir_data'])))
    idx_before = None
    if len(last_hs_saved) > 0: 
        start_datetime_available = []
        for ifile, filename in enumerate(last_hs_saved): 
            match = re.search(r'hotspots-(\d{4}-\d{2}-\d{2})_(\d{4})', os.path.basename(filename))
            date_part, time_part = match.groups()
            # Combine into a datetime object and set to UTC
            start_datetime_available.append( datetime.strptime(f"{date_part} {time_part}", "%Y-%m-%d %H%M").replace(tzinfo=timezone.utc) )
       
        idx = np.where(np.array(start_datetime_available)<=start_datetime)
        if len(idx[0])> 0: 
            idx_before = idx[0].max()
        
    #load last active fire saved as well as the past event. past event older than 7 days are not loaded
    if idx_before is not None:
        start_time_last_available = start_datetime_available[idx_before]
        if os.path.isfile("{:s}/hotspots-{:s}.gpkg".format(params['event']['dir_data'], start_time_last_available.strftime("%Y-%m-%d_%H%M") ) ):
            #activate loading existing data
            try:
                hsgdf_all_raw = gpd.read_file("{:s}/hotspots-{:s}.gpkg".format(params['event']['dir_data'], start_time_last_available.strftime("%Y-%m-%d_%H%M")) )
            except: 
                pdb.set_trace()
            fireEvents_files = sorted(glob.glob( params['event']['dir_data']+'/Pickles_{:s}_{:s}/*.pkl'.format('active', start_time_last_available.strftime("%Y-%m-%d_%H%M"))))
            ii = 0
            for id_, event_file in enumerate(fireEvents_files):
                event = fireEvent.load_fireEvent(event_file)
                while ii < event.id_fire_event:
                    fireEvents.append(None)
                    ii += 1
                fireEvents.append( fireEvent.load_fireEvent(event_file) ) 
                ii += 1
            fireEvent.Event._id_counter = fireEvents[-1].id_fire_event +1


            # Set your directory and threshold date
            directory_past = params['event']['dir_data']+'/Pickles_{:s}/'.format('past')
            threshold = start_datetime.replace(tzinfo=None) - timedelta(days=7) 
            
            if flag_PastFireEvent: 
                if os.path.isdir(directory_past):
                    filenames = [f for f in os.listdir(directory_past) if f.endswith(".pkl")]
                    # Create DataFrame
                    df = pd.DataFrame(filenames, columns=["filename"])
                    # Extract datetime from filename
                    df["datetime"] = pd.to_datetime(
                        df["filename"].str.extract(r"_(\d{4}-\d{2}-\d{2}_\d{4})")[0],
                        format="%Y-%m-%d_%H%M"
                    )
                    df["fire_id"] = df["filename"].str.extract(r"^(\d+)")
                    duplicates = df[df["fire_id"].duplicated(keep=False)]

                    if len(duplicates)> 0 : pdb.set_trace()

                    # Filter by threshold datetime
                    threshold = np.datetime64(threshold)
                    df_filtered = df[df["datetime"] > threshold]

                    # Compile regex pattern to extract datetime from filename
                    for ifile, event_file in enumerate(df_filtered['filename']):
                        #print(f"{100*ifile/len(df_filtered):.02f} {event_file}")
                        pastFireEvents.append( fireEvent.load_fireEvent(directory_past+event_file) ) 
   

            #pastFireEvents_files = sorted(glob.glob( params['event']['dir_data']+'/Pickles_{:s}/*.pkl'.format('past',)))
            #for id_, event_file in enumerate(pastFireEvents_files):
            #    print(f"{100*id_/len(pastFireEvents_files):.02f} {os.path.basename(event_file)}")
            #    try: 
            #        end_time_ = datetime.strptime( '_'.join(event_file.split('.')[0].split('_')[-2:]), "%Y-%m-%d_%H%M").replace(tzinfo=timezone.utc)
            #        if end_time_ > start_datetime - timedelta(days=7):
            #            pastFireEvents.append( fireEvent.load_fireEvent(event_file) ) 
            #    except: 
            #        pdb.set_trace()
       
        #date_now = date_now + timedelta(hours=1)    

    date_now = start_datetime + timedelta(hours=1)
    end_datetime = date_now + timedelta(hours=1)
    #print('')
    #print('init list Events: ', count_not_none(fireEvents), len(pastFireEvents))
    idate = 0
    flag_get_new_hs = False
    while date_now<end_datetime:
        print(date_now.strftime("%Y-%m-%d_%H%M"), end=' | ')
        

        #if len(fireEvents) == 0: 
        #    '''
        #    we load all hs from previous day and initialise event from there
        #    '''
        #    flag_init = True
        #    hour='*' #all
        #    previous_day = date_now - timedelta(days=1)    # subtract one day
        #    day = previous_day.strftime('%Y-%m-%d')        # convert back to string
        #else: 
        day = date_now.strftime('%Y-%m-%d')        # convert back to string
        hour = date_now.strftime('%H%M')
        #load last obs
        hsgdf = hstools.load_hs4lastObsAllSat(day,hour,params)
        if len(hsgdf)==0: 
            print('skip  ')
            date_now = date_now + timedelta(hours=1)    
            idate+=1
            continue
        
        # Compute time difference in seconds
        if date_now < hsgdf.timestamp.max().tz_localize('UTC'):
            delta_seconds = (hsgdf.timestamp.max().tz_localize('UTC')-date_now).total_seconds()
            sign_delta_seconds = 1
        else: 
            delta_seconds = (date_now - hsgdf.timestamp.max().tz_localize('UTC')).total_seconds()
            sign_delta_seconds = -1
        
        # Convert to hours and minutes
        hours = int(delta_seconds // 3600)
        minutes = int((delta_seconds % 3600) // 60)
        string_sign = "+" if sign_delta_seconds >= 0 else "-"
        print(f'\u0394t[h:m]= {string_sign} {hours:02d}:{minutes:02d} ', end=' | ')
        #print(hsgdf.timestamp.max().tz_localize('UTC'), end=' |  ')

        #filter HS industry
        n_hs_before_filter = len(hsgdf)
        hsgdf = filter_points_by_mask(hsgdf, maskHS_da )
        if len(hsgdf)==0: 
            print(' skip  ')
            date_now = date_now + timedelta(hours=1)    
            idate+=1
            continue
        n_hs_after_filter = len(hsgdf)
        print(f'- {n_hs_before_filter-n_hs_after_filter:04d} hs  |  ', end ='')
        flag_get_new_hs = True

        if hsgdf_all_raw is None:
            hsgdf_all_raw = hsgdf.copy()
        else: 
            duplicates = pd.merge(hsgdf_all_raw.assign(version=hsgdf_all_raw['version'].astype(str)), hsgdf.assign(version=hsgdf_all_raw['version'].astype(str)), how='inner')
            if len(duplicates)>0: 
                print('')
                print('find duplicate hotspot in new entry')
                pdb.set_trace()
            with warnings.catch_warnings(): 
                warnings.simplefilter("error", FutureWarning)
                try:
                    hsgdf.index = range(hsgdf_all_raw.index.max() + 1, hsgdf_all_raw.index.max() + 1 + len(hsgdf))
                    hsgdf_all_raw = pd.concat([hsgdf_all_raw.assign(version=hsgdf_all_raw['version'].astype(str)),hsgdf.assign(version=hsgdf_all_raw['version'].astype(str))])
                except: 
                    pdb.set_trace()
            
            #try:
            #    hsgdf_all_raw = hsgdf_all_raw.drop(index=19604)
            #except: 
            #    pass

        hsgdf_all = hsgdf_all_raw.copy()
        # Use DBSCAN for spatial clustering (define 1000 meters as the spatial threshold)
        # DBSCAN requires the distance in degrees, so you might need to convert meters to degrees.
        # Approximation: 1 degree latitude ~ 111 km.
        #epsilon = 1. / 111.0  # Approx. 1 km in degrees
        #db = DBSCAN(eps=epsilon, min_samples=1, metric='haversine').fit(np.array(gdf[['longitude','latitude' ]]))
        epsilon = 800  # Approx. 1 km in degrees
        hsgdf_all["x"] = hsgdf_all.geometry.x
        hsgdf_all["y"] = hsgdf_all.geometry.y
        db = DBSCAN(eps=epsilon, min_samples=1, metric='euclidean').fit(np.array(hsgdf_all[['x','y' ]]))

        # Add cluster labels to the GeoDataFrame
        hsgdf_all.loc[:,['cluster']] = db.labels_

        # Optionally, sort by time and perform temporal aggregation within each spatial cluster
        hsgdf_all = hsgdf_all.sort_values(by=['cluster', 'timestamp'])

        # You can define a time threshold, e.g., 24 hours
        time_threshold = pd.Timedelta('7 day')

        # Create a fire event ID by combining spatial clusters with temporal proximity
        hsgdf_all.loc[:,['fire_event']] = np.array((hsgdf_all.groupby('cluster')['timestamp'].apply(lambda x: (x.diff() > time_threshold).cumsum()+1 )))

        # Create a unique fire_event ID across all clusters by concatenating cluster and fire_event
        hsgdf_all.loc[:,['cluster_event_id']] = hsgdf_all['cluster'].astype(str) + '_' + hsgdf_all['fire_event'].astype(str)

        # Map the unique combinations to a continuous global fire event index
        hsgdf_all.loc[:,['cluster_fire_event']] = hsgdf_all['cluster_event_id'].factorize()[0]

        if 'original_index' in hsgdf_all.keys():
            hsgdf_all = hsgdf_all.drop(columns=['original_index'])
        #hsgdf_all = hsgdf_all.reset_index().rename(columns={'index': 'original_index'})  # If your index isn't already a column
        hsgdf_all['original_index'] = hsgdf_all.index

        # Aggregate hotspots by fire event
        fireCluster = hsgdf_all.groupby('cluster_fire_event').agg(
            total_hotspots=('latitude', 'count'),
            y=('y', 'mean'),
            x=('x', 'mean'),
            start_time=('timestamp', 'min'),
            end_time=('timestamp', 'max'),
            frp=('frp', 'sum'),  # Assuming FRP (Fire Radiative Power) is present
            indices_hs=('original_index', lambda x: list(x))  # Collect indices into a list
        ).reset_index()

        fireCluster.loc[:,['duration']] = (fireCluster['end_time']-fireCluster['start_time']).dt.total_seconds()/(24*3600) 
        fireCluster['center'] = [Point(xx,yy) for xx,yy in zip( fireCluster.x, fireCluster.y ) ]

        # Convert to GeoDataFrame
        geometry = [Point(xy) for xy in zip(fireCluster['x'], fireCluster['y'])]
        fireCluster = gpd.GeoDataFrame(fireCluster, geometry=geometry)
        fireCluster = fireCluster.set_crs(params['general']['crs'])
        
        #fireCluster = fireCluster.to_crs(params['general']['crs'])
        #fireCluster.plot()
        #plt.show()
        #pdb.set_trace()

        # Store alpha shape geometries here
        alpha_shapes = []

        # Choose a value for alpha. Smaller => tighter shape. You can tune this.
        alpha = 0.001  # Try different values

        # Iterate through each fire event cluster
        for event_id, group in hsgdf_all.groupby('cluster_fire_event'):
            group = group.drop_duplicates(subset=["latitude", "longitude"],keep='first')
            points = [(pt.x, pt.y) for pt in group.geometry]
            
            if len(points) <= 4:
                # Not enough points to compute alpha shape; fall back to convex hull or skip
                shape = MultiPoint(points).convex_hull
            else:
                shape = alphashape.alphashape(points, alpha)
                if (shape.is_empty): 
                    if (len(points)<10): 
                        shape = MultiPoint(points).convex_hull
                    else: 
                        print('pb in alphashape -- empty geom geenrated')

            alpha_shapes.append({
                'cluster_fire_event': event_id,
                'geometry': shape
            })

        # Create GeoDataFrame of alpha shapes per event
        alpha_shape_gdf = gpd.GeoDataFrame(alpha_shapes, crs=hsgdf_all.crs)
        fireCluster_ctr = alpha_shape_gdf
       
        if len(fireEvents) == 0: 
            #if no fire event were initialized, we set all cluster as fire event
            print(' create ', end=' |')
            for (_,cluster), (_,ctr) in zip(fireCluster.iterrows(),fireCluster_ctr.iterrows()):
                event = fireEvent.Event(cluster,ctr,fireCluster.crs,hsgdf_all_raw) 
                fireEvents.append(event)
                if event.id_fire_event == 242: pdb.set_trace()

        else: 
            print(' append ', end=' |')
            gdf_activeEvent = create_gdf_fireEvents(fireEvents)
           
            #here we go over each cluster and assign it to an existing event if its center is inside an active fire event. if not, this is a new one
            for (_,cluster), (_,ctr) in zip(fireCluster.iterrows(),fireCluster_ctr.iterrows()):
               
                #pdb.set_trace()
                #if cluster.end_time < date_now: 
                #    continue

                #if cluster.global_fire_event in gdf_activeEvent['global_fire_event']: continue
                   
                if (ctr.geometry.geom_type == 'Polygon') :
                    

                    #flag_found_matchingEvent = False
                    #for polygon in active fire event
                    #-------------
                    gdf_polygons = gdf_activeEvent[(gdf_activeEvent.geometry.type == 'Polygon')].copy()
                    gdf_polygons['intersection_ratio_area'] = gdf_polygons.geometry.intersection(ctr.geometry).area/gdf_polygons.area
                    active_poly_inside_cluster = gdf_polygons[gdf_polygons['intersection_ratio_area'] > 0.8 ].drop(columns=['intersection_ratio_area'])
                    
                    gdf_linepoints = gdf_activeEvent[(gdf_activeEvent.geometry.type == 'Point')|(gdf_activeEvent.geometry.type == 'LineString')].copy()
                    cluster_polygons_ =  gpd.GeoSeries([ctr.geometry],crs=gdf_linepoints.crs)
                    active_points_inside_cluster = gdf_linepoints[gdf_linepoints.geometry.apply(lambda point: cluster_polygons_.iloc[0].contains(point))].copy()

                    active_pp_matching_cluster = pd.concat([active_poly_inside_cluster,active_points_inside_cluster])

                    #print('\n########')
                    #print(len(active_pp_matching_cluster))

                    if len(active_pp_matching_cluster)==1:
                        idx_event = active_pp_matching_cluster.index[0]
                        fireEvents[idx_event].add(cluster,ctr,fireCluster.crs,hsgdf_all_raw)
                        gdf_activeEvent = create_gdf_fireEvents(fireEvents)
                        #flag_found_matchingEvent = True
                        continue
                    
                    elif len(active_pp_matching_cluster)>1:
                        #select the best option where to add the cluster depending of geometry in active_pp_matching_cluster
                        if 'Polygon' in active_pp_matching_cluster.geometry.geom_type.values:
                            idx_event = active_pp_matching_cluster.area.idxmax()
                        else: 
                            if 'LineString' in  active_pp_matching_cluster.geometry.geom_type: pdb.set_trace()
                            active_pp_matching_cluster['distancetopoly'] = active_pp_matching_cluster.geometry.distance(ctr.geometry.centroid).copy()
                            idx_event = active_pp_matching_cluster['distancetopoly'].idxmin()
                        
                        fireEvents[idx_event].add(cluster,ctr,fireCluster.crs,hsgdf_all_raw)

                        #merge with bigger
                        other_indices = active_pp_matching_cluster.index.difference([idx_event]).tolist()
                        #if 138 in  other_indices: pdb.set_trace()
                        for index_ in other_indices:
                            try:
                                fireEvents[idx_event].merge(fireEvents[index_])
                            except:
                                pdb.set_trace()
                        
                        fireEvents[idx_event].mergeWith(other_indices)
                        
                        #set merged event to past event
                        for index_ in other_indices:
                            #print('--', index_)
                            element = fireEvents[index_]
                            fireEvents[index_] = None
                            if flag_PastFireEvent: pastFireEvents.append(element)

                        gdf_activeEvent = create_gdf_fireEvents(fireEvents)
                        #flag_found_matchingEvent = True
                        continue


                if (ctr.geometry.geom_type == 'Point') | (ctr.geometry.geom_type == 'LineString'):

                    gdf_polygons = gdf_activeEvent[(gdf_activeEvent.geometry.type == 'Polygon')].copy()
                    active_gdf_polygons_containingPt = gdf_polygons[gdf_polygons.contains(ctr.geometry)]
                
                    gdf_linepoints = gdf_activeEvent[(gdf_activeEvent.geometry.type == 'Point')|(gdf_activeEvent.geometry.type == 'LineString')].copy()
                    active_linepoints_close2Pt = gdf_linepoints[ gdf_linepoints.distance(ctr.geometry) < 2.e3 ]

                    active_pp_matching_cluster = pd.concat([active_gdf_polygons_containingPt,active_linepoints_close2Pt])

                    if len(active_pp_matching_cluster)==1:
                        idx_event = active_pp_matching_cluster.index[0]
                        fireEvents[idx_event].add(cluster,ctr,fireCluster.crs,hsgdf_all_raw)
                        gdf_activeEvent = create_gdf_fireEvents(fireEvents)
                        #flag_found_matchingEvent = True
                        continue
                    
                    elif len(active_pp_matching_cluster)>1:
                        #select the best option where to add the cluster depending of geometry in active_pp_matching_cluster
                        if 'Polygon' in active_pp_matching_cluster.geometry.geom_type.values:
                            idx_event = active_pp_matching_cluster.area.idxmax()
                        else: 
                            if 'LineString' in  active_pp_matching_cluster.geometry.geom_type: pdb.set_trace()
                            active_pp_matching_cluster['distancetopoly'] = active_pp_matching_cluster.geometry.distance(ctr.geometry.centroid).copy()
                            idx_event = active_pp_matching_cluster['distancetopoly'].idxmin()
                        fireEvents[idx_event].add(cluster,ctr,fireCluster.crs,hsgdf_all_raw)

                        #merge with bigger
                        other_indices = active_pp_matching_cluster.index.difference([idx_event]).tolist()
                        #if 138 in  other_indices: pdb.set_trace()
                        for index_ in other_indices:
                            fireEvents[idx_event].merge(fireEvents[index_])
                        
                        fireEvents[idx_event].mergeWith(other_indices)
                        
                        #set merged event to past event
                        for index_ in other_indices:
                            #print('--', index_)
                            element = fireEvents[index_]
                            fireEvents[index_] = None
                            if flag_PastFireEvent: pastFireEvents.append(element)

                        gdf_activeEvent = create_gdf_fireEvents(fireEvents)
                        #flag_found_matchingEvent = True
                        continue

                #if we are here, we have a new event
                #if 19604 in cluster.indices_hs: pdb.set_trace()
                #if 14634 in cluster.indices_hs: pdb.set_trace()
                #if cluster.frp in gdf_activeEvent['frp'].values: pdb.set_trace()
                #print('????????????????? new event')
                new_event = fireEvent.Event(cluster,ctr,fireCluster.crs, hsgdf_all_raw) 
                fireEvents.append(new_event)
                #pdb.set_trace()
                if new_event.id_fire_event == 242: pdb.set_trace()

        #remove fireEvent that were updated more than two day ago. 
        for id_, event in enumerate(fireEvents):
            if event is None: continue
            if event.times[-1] < (pd.Timestamp(date_now) - timedelta(days=2)):
                element = fireEvents[id_]
                fireEvents[id_] = None
                if flag_PastFireEvent: pastFireEvents.append(element)
        gdf_activeEvent = create_gdf_fireEvents(fireEvents)
        
        #remove old hotspot older than 7 days
        dt_naive = (date_now - timedelta(days=7)).replace(tzinfo=None)
        date_now_64 = pd.Timestamp(dt_naive)
        hsgdf_all_raw = hsgdf_all_raw[hsgdf_all_raw['timestamp']>=date_now_64 ]

        #end temporal loop
        date_now = date_now + timedelta(hours=1)    # subtract one day
        idate+=1

    date_now = date_now - timedelta(hours=1)    # subtract one day
    
    #save 
    #clean fire event dir
    #if os.path.isdir(params['event']['dir_data']+'Pickles_active/'):
    #    shutil.rmtree(params['event']['dir_data']+'Pickles_active/')
 
    #print(flag_get_new_hs, end_datetime)
    if flag_get_new_hs:
        if len(fireEvents)>0:
            gdf_activeEvent = create_gdf_fireEvents(fireEvents)
            #gdf_to_gpkgfile(gdf_activeEvent, params, end_datetime, 'firEvents')
            gdf_to_geojson(gdf_activeEvent.to_crs(4326), params, date_now, 'firEvents')
            gdf_to_gpkgfile(hsgdf_all_raw, params, date_now, 'hotspots')
            
        for id_, event in enumerate(fireEvents):
            if event is not None: 
                event.save( 'active', params, date_now)

        if flag_PastFireEvent:
            print('  FireEvents saved: active: {:6d}  past: {:6d}'.format(count_not_none(fireEvents), len(pastFireEvents)))
        else:
            print('  FireEvents saved: active: {:6d} '.format(count_not_none(fireEvents), ))

    for id_, event in enumerate(pastFireEvents):
        event.save( 'past', params)



    return end_datetime, fireEvents, pastFireEvents

##############################################
def gdf_to_gpkgfile(gdf_activeEvent, params, datetime_, name_):
    tmp_path = "./{}-{}.gpkg".format(name_, datetime_.strftime("%Y-%m-%d_%H%M"))
    gdf_activeEvent.to_file(tmp_path, driver="GPKG")
    # Move to mounted share
    dst_path = os.path.join(params['event']['dir_data'], os.path.basename(tmp_path))
    shutil.move(tmp_path, dst_path)
    return None

##############################################
def gdf_to_geojson(gdf_activeEvent, params, datetime_, name_):
    tmp_path = "./{:s}-{:s}.geojson".format(name_, datetime_.strftime("%Y-%m-%d_%H%M"))
    #set time attribute to time of the fire event and time_obs to the time of the last obs
    #gdf_activeEvent = gdf_activeEvent.rename(columns={'time': 'time_obs'})
    #gdf_activeEvent['time'] = np.datetime64(datetime_)
    
    gdf_activeEvent.to_file(tmp_path, driver="GeoJSON")
    # Move to mounted share
    dst_path = os.path.join(params['event']['dir_geoJson'], os.path.basename(tmp_path))
    shutil.move(tmp_path, dst_path)
    return None

###############################################
def count_not_none(lst):
    return sum(1 for item in lst if item is not None)


###############################################
def plot(params, date_now, fireEvents, pastFireEvents, flag_plot_hs=True, flag_remove_singleHs=False):
    # Create a figure with Cartopy
    fig, ax = plt.subplots(figsize=(10, 6),
                       subplot_kw={'projection': ccrs.epsg(params['general']['crs']) })  # PlateCarree() == EPSG:4326

    # Add basic map features
    ax.coastlines(linewidth=0.2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.1)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

    for event in fireEvents:
        if event is None: continue
        #if event.id_fire_event in [102]: continue
        times_numeric = pd.to_datetime(event.times).astype(int) / 10**9  # Convert to seconds since epoch

        #if len(event.times)> 0: pdb.set_trace()

        # Normalize the time values to 0-1 for colormap
        norm = mcolors.Normalize(vmin=times_numeric.min(), vmax=times_numeric.max())
        cmap = mpl.colormaps.get_cmap('jet')

        # Map each time value to a color
        colors = [cmap(norm(t)) for t in times_numeric]
       
        #event.plot(ax=ax,alpha=0.4,c='k',markersize=10)
        for (irow,row_ctr), (_,row_hs) in zip(event.ctrs.iterrows(),event.hspots.iterrows()):
            #if mm : 
            #    fig, ax = plt.subplots(figsize=(10, 6),
            #           subplot_kw={'projection': ccrs.epsg(params['general']['crs']) })  # PlateCarree() == EPSG:4326

            if row_ctr.geometry.geom_type == 'Point':
                if flag_remove_singleHs: 
                    if len(event.times) == 1: continue
                gpd.GeoSeries([row_ctr.geometry]).plot(ax=ax, color=colors[irow], linewidth=0.1, zorder=2, alpha=0.7, markersize=1)
            else:
                gpd.GeoSeries([row_ctr.geometry]).plot(ax=ax, facecolor='none',edgecolor=colors[irow], cmap=cmap, linewidth=1, zorder=2, alpha=0.7)
            if flag_plot_hs:
                points = gpd.GeoSeries([row_hs.geometry]).explode(index_parts=False)
                points.plot(ax=ax, color=colors[irow], alpha=0.5, markersize=40)
    
    for event in pastFireEvents:
        if flag_remove_singleHs: 
            if len(event.times) == 1: continue
        event.ctrs.plot(ax=ax, facecolor='none',edgecolor='k', alpha=0.2, linewidth=0.1, linestyle='--', zorder=1, markersize=1)

    ax.set_title(date_now)

    # Set extent if needed
    extent=params['general']['domain'].split(',')
    ax.set_extent([extent[i] for i in [0,2,1,3]])

    # Add gridlines with labels
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
   
    filename = "{:s}/Fig/fireEvent-{:s}-{:s}.png".format(params['event']['dir_data'],params['general']['domainName'],date_now.strftime("%Y-%m-%d_%H%M")) 
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    fig.savefig(filename,dpi=600)
    plt.close(fig)

    #plt.show()


#############################
if __name__ == '__main__':
#############################
    
    '''
    SILEX
    domain: -10,35,20,46
    '''
    importlib.reload(hstools)
    src_dir = os.path.dirname(os.path.abspath(__file__))
  
    parser = argparse.ArgumentParser(description="fireEventTracking")
    parser.add_argument("--inputName", type=str, help="name of the configuration input", )
    parser.add_argument("--log_dir", type=str, help="Directory for logs", default='/mnt/dataEstrella2/SILEX/VIIRS-HotSpot/FireEvents/log/')

    args = parser.parse_args()

    inputName = args.inputName
    log_dir = args.log_dir
   
    #log_dir = sys.argv[2]
    #inputName = sys.argv[1]

    #init dir
    if inputName == '202504': 
        params = init('202504',log_dir) 
        start = datetime.strptime('2025-04-12_0000', '%Y-%m-%d_%H%M')
        end = datetime.strptime('2025-04-15_0000', '%Y-%m-%d_%H%M')
    
    elif inputName == 'AVEIRO': 
        params = init('AVEIRO',log_dir) 
        start = datetime.strptime('2024-09-15_0000', '%Y-%m-%d_%H%M')
        end = datetime.strptime('2024-09-20_2300', '%Y-%m-%d_%H%M')
    
    elif inputName == 'SILEX': 
        params = init('SILEX',log_dir)
        if os.path.isfile(log_dir+'/timeControl.txt'): 
            with open(log_dir+'/timeControl.txt','r') as f:
                start = datetime.strptime(f.readline().strip(), '%Y-%m-%d_%H%M').replace(tzinfo=timezone.utc)
        else:
            start = datetime.strptime(params['event']['start_time'], '%Y-%m-%d_%H%M').replace(tzinfo=timezone.utc)
        
        end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
  
    else:
        print('missing inputName')
        sys.exit()
    
    #make sure all time are UTC
    start = start.replace(tzinfo=timezone.utc)
    end   = end.replace(tzinfo=timezone.utc)
    print('########times#########')
    print(start)
    print(end)
    print('######################')


    # Loop hourly
    current = start
    end_time = None
    while current <= end:

        end_time = current  
        #get last time processed
        if os.path.isfile("{:s}/hotspots-{:s}.gpkg".format(params['event']['dir_data'],(current+timedelta(hours=1)).strftime("%Y-%m-%d_%H%M")) ):
            print(current, end=' already done\n')
            current += timedelta(hours=1)
            continue 
        
        #track perimeter
        start_datetime = current.strftime('%Y-%m-%d_%H%M')
        #end_datetime   = (current + timedelta(hours=1)).strftime('%Y-%m-%d_%H%M')
        date_now, fireEvents, pastFireEvents = perimeter_tracking(params, start_datetime)#,end_datetime)
        
        if date_now.hour == 20 and date_now.minute == 0:
            #ploting
            plot(params, date_now, fireEvents, pastFireEvents, flag_plot_hs=False, flag_remove_singleHs=True)

        #control hourly loop
        current += timedelta(hours=1)

    with open(log_dir+'/timeControl.txt','w') as f:
        f.write(end_time.strftime('%Y-%m-%d_%H%M'))

