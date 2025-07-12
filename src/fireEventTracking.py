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
from shapely.geometry import Point, MultiPoint, Polygon, MultiPolygon
from shapely.ops import unary_union
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
from concurrent.futures import ThreadPoolExecutor
from pyproj import Transformer
import tempfile
from multiprocessing import Pool
import polyline 
import requests 
import json 
import socket

warnings.filterwarnings("error", category=pd.errors.SettingWithCopyWarning)

#home brewed
import hstools
import fireEvent
import os
import subprocess
import discordMessage

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
def create_gdf_fireEvents(params,fireEvents):
    
    '''
    filtered_fireEvents = [x for x in fireEvents if x is not None]
    
    gdf_activeEvent = pd.concat( [fireEvent.ctrs.iloc[-1:] for fireEvent in  filtered_fireEvents] ).reset_index().drop(columns=['index'])

    gdf_activeEvent['time'] = [fireEvent.times[-1] for fireEvent in  filtered_fireEvents]
    gdf_activeEvent['center'] = [fireEvent.centers[-1] for fireEvent in  filtered_fireEvents] 
    gdf_activeEvent['frp'] = [fireEvent.frps[-1] for fireEvent in  filtered_fireEvents] 
    gdf_activeEvent['id_fire_event'] = [fireEvent.id_fire_event for fireEvent in  filtered_fireEvents]
    gdf_activeEvent = gdf_activeEvent.set_index('id_fire_event')

    eurostat_postcode_boundaries = gpd.read_file(params['general']['root_data']+'/'+params['event']['eurostat'])
    gdf_activeEvent = assign_fire_names_from_postcode(gdf_activeEvent, eurostat_postcode_boundaries)
    '''
        
    # Filter out None values
    filtered_fireEvents = [x for x in fireEvents if x is not None]

    if len(filtered_fireEvents)>0:
        crs = filtered_fireEvents[0].ctrs.crs
    else: 
        crs = params['general']['crs']
    # Extract data
    geometries = []
    times = []
    centers = []
    frps = []
    ids = []
    names = []
    ffmnhUrl = []
    for fe in filtered_fireEvents:
        last_ctr = fe.ctrs.iloc[-1]
        geometries.append(last_ctr.geometry)
        times.append(fe.times[-1])
        centers.append(fe.centers[-1])
        frps.append(fe.frps[-1])
        ids.append(fe.id_fire_event)
        names.append(fe.fire_name)
        try:
            ffmnhUrl.append(fe.ffmnhUrl)
        except: 
            ffmnhUrl.append('none')

    # Construct GeoDataFrame
    if crs != None:
        gdf_activeEvent = gpd.GeoDataFrame({
            'time': times,
            'center': centers,
            'frp': frps,
            'id_fire_event': ids,
            'name': names,
            'ffmnhUrl': ffmnhUrl,
            'geometry': geometries
        }, geometry='geometry', crs=crs)
    else:
        gdf_activeEvent = gpd.GeoDataFrame({
            'time': times,
            'center': centers,
            'frp': frps,
            'id_fire_event': ids,
            'name': names,
            'ffmnhUrl': ffmnhUrl,
            'geometry': geometries
        }, geometry='geometry')

    
    gdf_activeEvent = gdf_activeEvent.set_index('id_fire_event')

    return gdf_activeEvent


####################################################
def assign_fire_names_from_postcode(
    gdf_fire: gpd.GeoDataFrame,
    gdf_postcode: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Assigns fire_name to fire events using spatial join with postcode and city info.
    Adds only the 'fire_name' column to the original GeoDataFrame.

    Parameters
    ----------
    gdf_fire : GeoDataFrame
        Fire events with 'center' (Point geometry) and 'time' columns.
    gdf_postcode : GeoDataFrame
        Postcode boundaries with at least: 'geometry', 'POSTCODE', 'CITY', 'CNTR_CODE'.
    crs_postcode : str
        CRS of the postcode GeoDataFrame (default 'EPSG:25829').

    Returns
    -------
    GeoDataFrame
        Input GeoDataFrame with one additional column: 'fire_name'.
    """
    # Make a copy to avoid modifying original
    gdf_fire2 = gdf_fire.copy()

    # Ensure proper CRS
    gdf_postcode = gdf_postcode.to_crs(gdf_fire.crs)
    gdf_fire2 = gdf_fire2.set_geometry("center").set_crs(gdf_fire.crs)

    # Spatial join
    joined = gpd.sjoin(
        gdf_fire2[["center", "time"]],
        gdf_postcode[["geometry", "NSI_CODE", "COMM_NAME", "CNTR_CODE", "NUTS_CODE"]],
        how="left",
        predicate="intersects"
    )

    # Compose fire_name only
    fire_name = (
        "fire_" +
        joined["CNTR_CODE"].fillna("XX") + "_" +
        joined["NSI_CODE"].fillna("00000").astype(str) + "_" +
        joined["COMM_NAME"].replace(' ','').fillna("UnknownCity").str.replace(" ", "") + "_" +
        joined["NUTS_CODE"].fillna("00000").astype(str) + "_" +
        pd.to_datetime(joined["time"]).dt.strftime("%Y%m%d")
    )

    # Insert only fire_name back into the original dataframe
    gdf_fire["fire_name"] = fire_name.values

    return gdf_fire


####################################################
def is_mounted(path):
    return os.path.ismount(path)

####################################################
def init(config_name, sensorName, log_dir):
    script_dir = Path(__file__).resolve().parent
    params = hstools.load_config(str(script_dir)+f'/../config/config-{config_name}.yaml')
 
    if params['general']['use_sedoo_drive']:
        if not ( is_mounted(params['general']['root_data'])) :
            print('missing AERIS disc')
            sys.exit()
    
    params['hs']['dir_data'] = params['general']['root_data'] + params['hs']['dir_data'].replace('ORIGIN',sensorName)
    params['event']['dir_data'] = params['general']['root_data'] + params['event']['dir_data'].replace('ORIGIN',sensorName)
    params['general']['sensor'] = sensorName

    if params['general']['sensor'] == 'FCI':
        params['general']['discord_channel']= 'silex-fire-alert-fci'   #int(os.environ['discord_channel_id_fire_alert_fci'])
    elif params['general']['sensor'] == 'VIIRS':
        params['general']['discord_channel']= 'silex-fire-alert-viirs' #int(os.environ['discord_channel_id_fire_alert_viirs'])

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
        params['event']['dir_geoJson'] = params['general']['root_data'] + params['event']['dir_geoJson'].replace('ORIGIN',sensorName)
        os.makedirs(params['event']['dir_geoJson'],exist_ok=True)
    if 'dir_frp' in params['event'].keys():
        params['event']['dir_frp'] = params['general']['root_data'] + params['event']['dir_frp'].replace('ORIGIN',sensorName)
        os.makedirs(params['event']['dir_frp'],exist_ok=True)
    

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
def cache_file(params, fireEvents_files, max_workers=16):
    local_dir = tempfile.mkdtemp()

    def copy_to_local(remote_file):
        local_path = os.path.join(local_dir, os.path.basename(remote_file))
        if not os.path.exists(local_path):
            try:
                shutil.copy(remote_file, local_path)
            except Exception as e:
                print(f"Warning: Failed to copy {remote_file}: {e}")
        return local_path

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        local_paths = list(executor.map(copy_to_local, fireEvents_files))

    return local_paths, local_dir
            

####################################################
def post_on_discord_and_runffMNH(params,event):
    
    if socket.gethostname() == 'pc70852': return None
    
    if 'fire_FR' not in event.fire_name:
        event.add_ffmnhSimu('none')
        return None
    transformer = Transformer.from_crs("EPSG:{:d}".format(params['general']['crs']), "EPSG:4326", always_xy=True)
    bell = "\U0001F514"

    pt = event.centers[-1]
    x, y = pt.x, pt.y 
    lon, lat = transformer.transform(x,y)
    ffmnh_info = run_ffMNH_corte(params, event, lon, lat)
    if ffmnh_info is not None:
        url = f"https://forefire.univ-corse.fr/firecast/{ffmnh_info['path']}"
        loc = f"Location: Lon: {lon:.6f}, Lat: {lat:.6f}\n\t\tForeFireMNH Simulation: {url}"
        event.add_ffmnhSimu(url)
    else:
        url = f"https://www.openstreetmap.org/?mlat={lat:.6f}&mlon={lon:.6f}#map=14/{lat:.6f}/{lon:.6f}"
        loc = f"Location: Lon: {lon:.6f}, Lat: {lat:.6f}\n\t\tOpen Street Map loc: {url}"
        event.add_ffmnhSimu('none')
    
    discordMessage.send_message_to_discord_viaAeris(
                                            bell+f"FET: new Fire from {params['general']['sensor']}"+\
                                            f"\n\t\tfrp: {event.frps[-1]:.2f} MW"+\
                                            f'\n\t\ttime: {event.times[-1]}'+\
                                            f"\n\t\tpostcode-commune: {'-'.join(event.fire_name.split('_')[2:4])}"+\
                                            f'\n\t\tcenter: {loc}'+\
                                            f'\n\t\tarea: {1.e-4*event.areas[-1]:.2f} ha' , 
                                        params['general']['discord_channel']
                                                   )

    return ffmnh_info


####################################################
def run_ffMNH_corte(params, event, lon, lat):
   
    if socket.gethostname() == 'pc70852': return None
    url = "https://forefire.univ-corse.fr/simapi/forefireAPI.php"

    data = {
        'command': 'init',
        'path': f'ronanSilex{event.id_fire_event}',
        'coords': f'{polyline.encode([(lat,lon)])}',
        'ventX': '0',
        'ventY': '0',
        'date': event.times[-1].strftime('%Y-%m-%dT%H:%M:%SZ'),
        'model': 'Rothermel',
        'apikey': 'DEMO',
        'message': 'QUICK800'
    }
    #print(f'ronanSilex{event.id_fire_event}')
    #print(polyline.encode([(lat,lon)]))
    #print(event.times[-1].strftime('%Y-%m-%dT%H:%M:%SZ'))

    response = requests.post(url, data=data, verify=False)

    # Check the response
    #print("Status Code:", response.status_code)
    #print("Response Text:", response.text)
  
    if response.status_code == 200:
        try: 
            return json.loads(response.text)
        except:     
            return None
    else: 
        return None

####################################################
#def perimeter_tracking(params, start_datetime,end_datetime, flag_restart=False):
def perimeter_tracking(params, start_datetime, maskHS_da, dt_minutes):
    
    start_datetime = datetime.strptime(f'{start_datetime}', '%Y-%m-%d_%H%M' ).replace(tzinfo=timezone.utc)
    #end_datetime   = datetime.strptime(f'{end_datetime}', '%Y-%m-%d_%H%M').replace(tzinfo=timezone.utc)
    
    gdf_postcode = gpd.read_file(params['general']['root_data']+'/'+params['event']['eurostat'])

    '''
    if params['event']['file_HSDensity_ESAWorldCover'] != 'None': 
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
    #
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
            
            local_fireEvents_files, local_dir = cache_file(params, fireEvents_files)
            ii = 0
            for id_, event_file in enumerate(local_fireEvents_files):
                event = fireEvent.load_fireEvent(event_file)
                while ii < event.id_fire_event:
                    fireEvents.append(None)
                    ii += 1
                #fireEvents.append( fireEvent.load_fireEvent(event_file) ) 
                fireEvents.append( event ) 
                ii += 1
            if ii > 0:  
                fireEvent.Event._id_counter = fireEvents[-1].id_fire_event +1
            else: 
                pdb.set_trace()
            shutil.rmtree(local_dir)

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

    date_now     = start_datetime #+ timedelta(minutes=dt_minutes)
    end_datetime = start_datetime + timedelta(minutes=dt_minutes)
    if params['general']['sensor'] == 'VIIRS':
        dt_minutes_perimeters =  dt_minutes
    elif params['general']['sensor'] == 'FCI':
        dt_minutes_perimeters = 10. 
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
            print('skip [no hs]  ', end='\n')
            date_now = date_now + timedelta(minutes=dt_minutes_perimeters)    
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
        if maskHS_da is not None: hsgdf = filter_points_by_mask(hsgdf, maskHS_da )
        if len(hsgdf)==0: 
            print(' skip  [no hs after fixhs filter] ')
            date_now = date_now + timedelta(minutes=dt_minutes_perimeters)    
            idate+=1
            continue
        n_hs_after_filter = len(hsgdf)
        print(f'{n_hs_after_filter:06d} HS  |  ', end ='')
        print(f'- {n_hs_before_filter-n_hs_after_filter:04d} fixHS  |  ', end ='')
        flag_get_new_hs = True

        if hsgdf_all_raw is None:
            hsgdf_all_raw = hsgdf.copy()
        else: 
            if params['general']['sensor'] == 'VIIRS':
                duplicates = pd.merge(hsgdf_all_raw.assign(version=hsgdf_all_raw['version'].astype(str)), hsgdf.assign(version=hsgdf_all_raw['version'].astype(str)), how='inner')
            elif params['general']['sensor'] == 'FCI':
                duplicates = pd.merge(hsgdf_all_raw, hsgdf, how='inner')
            
            if len(duplicates)>0: 
                print('')
                print('find duplicate hotspot in new entry')
                pdb.set_trace()
            
            with warnings.catch_warnings(): 
                warnings.simplefilter("error", FutureWarning)
                try:
                    hsgdf.index = range(hsgdf_all_raw.index.max() + 1, hsgdf_all_raw.index.max() + 1 + len(hsgdf))
                    if params['general']['sensor'] == 'VIIRS':
                        hsgdf_all_raw = pd.concat([hsgdf_all_raw.assign(version=hsgdf_all_raw['version'].astype(str)),hsgdf.assign(version=hsgdf_all_raw['version'].astype(str))])
                    elif params['general']['sensor'] == 'FCI':
                        hsgdf_all_raw = pd.concat([hsgdf_all_raw,hsgdf])
                
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
        if params['general']['sensor'] == 'VIIRS':
            epsilon = 800  
        elif params['general']['sensor'] == 'FCI':
            epsilon = 2000  
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
            #frp=('frp', 'sum'),  # Assuming FRP (Fire Radiative Power) is present
            indices_hs=('original_index', lambda x: list(x))  # Collect indices into a list
        ).reset_index()

        #compute FRP per cluster using only the last hotspot
        frp_cluster = []
        for icluster, cluster in fireCluster.iterrows():
            idx_valid_hs = hsgdf_all.loc[fireCluster['indices_hs'][icluster]]['timestamp'] >= np.datetime64(date_now.replace(tzinfo=None))
            frp_ = hsgdf_all.loc[fireCluster['indices_hs'][icluster]].frp[idx_valid_hs].sum()
            frp_cluster.append(frp_)
        fireCluster['frp'] = frp_cluster

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
            
        #ensure poly are convex
        fireCluster_ctr['geometry'] = fireCluster_ctr['geometry'].apply(make_convex)
 
        #keep only cluster with at least one hs from dat_now
        idxs = fireCluster['end_time'] >= np.datetime64(date_now.replace(tzinfo=None))
        fireCluster =  fireCluster[idxs]
        fireCluster_ctr = fireCluster_ctr[idxs]

        if len(fireEvents) == 0: 
            #if no fire event were initialized, we set all cluster as fire event
            print(' create ', end=' |')
            for (_,cluster), (_,ctr) in zip(fireCluster.iterrows(),fireCluster_ctr.iterrows()):
                event = fireEvent.Event(cluster,ctr,fireCluster.crs,hsgdf_all_raw,gdf_postcode) 
                fireEvents.append(event)
                post_on_discord_and_runffMNH(params,event)
                #if event.id_fire_event == 242: pdb.set_trace()

        else: 
            print(' append ', end=' |')
            gdf_activeEvent = create_gdf_fireEvents(params,fireEvents)
           
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
                        gdf_activeEvent = create_gdf_fireEvents(params,fireEvents)
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
                            if flag_PastFireEvent: 
                                #for event in pastFireEvents:
                                #    if event.id_fire_event == 872: pdb.set_trace()
                                pastFireEvents.append(element)

                        gdf_activeEvent = create_gdf_fireEvents(params,fireEvents)
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
                        gdf_activeEvent = create_gdf_fireEvents(params,fireEvents)
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
                            if flag_PastFireEvent: 
                                #for event in pastFireEvents:
                                #    if event.id_fire_event == 872: pdb.set_trace()
                                pastFireEvents.append(element)

                        gdf_activeEvent = create_gdf_fireEvents(params,fireEvents)
                        #flag_found_matchingEvent = True
                        continue

                #if we are here, we have a new event
                #if 19604 in cluster.indices_hs: pdb.set_trace()
                #if 14634 in cluster.indices_hs: pdb.set_trace()
                #if cluster.frp in gdf_activeEvent['frp'].values: pdb.set_trace()
                #print('????????????????? new event')
                new_event = fireEvent.Event(cluster,ctr,fireCluster.crs, hsgdf_all_raw,gdf_postcode) 
                fireEvents.append(new_event)
                post_on_discord_and_runffMNH(params,new_event)
                #if new_event.id_fire_event == 242: pdb.set_trace()

        #remove fireEvent that were updated more than two day ago. 
        for id_, event in enumerate(fireEvents):
            if event is None: continue
            if event.times[-1] < (pd.Timestamp(date_now) - timedelta(days=2)):
                element = fireEvents[id_]
                fireEvents[id_] = None
                if flag_PastFireEvent: 
                    #for event in pastFireEvents:
                    #    if event.id_fire_event == 872: pdb.set_trace()
                    pastFireEvents.append(element)
       
        if len(fireEvents) == 0 : pdb.set_trace()

        gdf_activeEvent = create_gdf_fireEvents(params,fireEvents)
        
        #remove old hotspot older than 7 days
        dt_naive = (date_now - timedelta(days=7)).replace(tzinfo=None)
        date_now_64 = pd.Timestamp(dt_naive)
        hsgdf_all_raw = hsgdf_all_raw[hsgdf_all_raw['timestamp']>=date_now_64 ]

        #end temporal loop
        date_now = date_now + timedelta(minutes=dt_minutes_perimeters)    # subtract one day
        idate+=1
    
        if date_now<end_datetime : print('')

    #date_now = date_now - timedelta(minutes=dt_minutes_perimeters)    # subtract one day
    
    #save 
    #clean fire event dir
    #if os.path.isdir(params['event']['dir_data']+'Pickles_active/'):
    #    shutil.rmtree(params['event']['dir_data']+'Pickles_active/')
 
    #print(flag_get_new_hs, end_datetime)
    if flag_get_new_hs:
        if len(fireEvents)>0:
            gdf_activeEvent = create_gdf_fireEvents(params,fireEvents)
            #gdf_to_gpkgfile(gdf_activeEvent, params, end_datetime, 'firEvents')
            gdf_to_geojson(gdf_activeEvent.to_crs(4326), params, start_datetime, 'firEvents')
            gdf_to_gpkgfile(hsgdf_all_raw, params, start_datetime, 'hotspots')
        
        '''
        for id_, event in enumerate(fireEvents):
            if event is not None: 
                event.save( 'active', params, start_datetime)
                #print FRP time serie of active event 
                fig = plt.figure(figsize=(10,5))
                ax=plt.subplot(111)
                ax.plot(event.times, event.frps, marker='o', linestyle='-')
                ax.set_xlabel('time')
                ax.set_ylabel('FRP (MW)')
                fig.savefig(f"{params['event']['dir_frp']:s}/{event.id_fire_event:09d}.png",dpi=100)
                plt.close(fig)
        '''

        # Create a temporary directory for storing plots
        tmp_dir = tempfile.mkdtemp()

        # Collect tasks for parallel copy
        copy_tasks = []

        dir_Pkl = None
        # Loop over fire events
        for id_, event in enumerate(fireEvents):
            if event is not None:
                # Save metadata or internal state
                
                tmp_filePkl, dest_filePkl = event.save('active', params, start_datetime, local_dir=tmp_dir)
                copy_tasks.append((tmp_filePkl, dest_filePkl))
                if dir_Pkl == None: dir_Pkl = os.path.dirname(dest_filePkl)
                # Create and save the FRP time series plot in temporary dir
                mpl.rcdefaults()
                mpl.rcParams['axes.labelsize'] = 20.
                mpl.rcParams['legend.fontsize'] = 'small'
                mpl.rcParams['legend.fancybox'] = True
                mpl.rcParams['font.size'] = 20.
                mpl.rcParams['xtick.labelsize'] = 20.
                mpl.rcParams['ytick.labelsize'] = 20.
                mpl.rcParams['figure.subplot.left'] = .07
                mpl.rcParams['figure.subplot.right'] = .95
                mpl.rcParams['figure.subplot.top'] = .91
                mpl.rcParams['figure.subplot.bottom'] = .07
                mpl.rcParams['figure.subplot.hspace'] = 0.05
                mpl.rcParams['figure.subplot.wspace'] = 0.05  

                fig = plt.figure(figsize=(10, 5))
                ax = plt.subplot(111)
                ax.plot(event.times, event.frps, marker='o', linestyle='-')
                ax.set_xlabel('time')
                ax.set_ylabel('FRP (MW)')
                ax.set_title(' '.join(event.fire_name.split('_')[2:4]))
                # File paths
                tmp_file = os.path.join(tmp_dir, f"{event.id_fire_event:09d}.png")
                dest_file = os.path.join(params['event']['dir_frp'], f"{event.id_fire_event:09d}.png")
                
                # Save figure and close
                fig.savefig(tmp_file, dpi=100)
                plt.close(fig)

                # Add to copy task list
                copy_tasks.append((tmp_file, dest_file))
      
        if dir_Pkl is not None:
            os.makedirs(dir_Pkl, exist_ok=True)
            # Parallel copy using multiprocessing
            with Pool(processes=16) as pool:
                pool.map(copy_file, copy_tasks)

        # Optional cleanup (uncomment to remove temp dir after copy)
        shutil.rmtree(tmp_dir)
    

        if flag_PastFireEvent:
            print('  FireEvents saved: active: {:6d}  past: {:6d}'.format(count_not_none(fireEvents), len(pastFireEvents)))
        else:
            print('  FireEvents saved: active: {:6d} '.format(count_not_none(fireEvents), ))

    for id_, event in enumerate(pastFireEvents):
        event.save( 'past', params)



    return start_datetime, fireEvents, pastFireEvents


###############################################
def make_convex(geom):
    if isinstance(geom, Polygon):
        return geom.convex_hull
    elif isinstance(geom, MultiPolygon):
        # Combine all parts then get the convex hull
        return unary_union(geom).convex_hull
    else:
        return geom  # Leave Point, LineString, etc., unchanged


###############################################
# Function to copy a single file (for multiprocessing)
def copy_file(task):
    src, dst = task
    shutil.copy2(src, dst)

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
   
    sensor = params['general']['sensor']

    url_image_root = "https://api.sedoo.fr/aeris-euburn-silex-rest/resource/fires/{:s}/fire_events/FRP/{:09d}.png"

    gdf_activeEvent['image'] = gdf_activeEvent.index.to_series().apply(
                                                                        lambda x: url_image_root.format(sensor, int(x))
                                                                     )

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
    mpl.rcdefaults()
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['legend.fontsize'] = 'small'
    mpl.rcParams['legend.fancybox'] = True
    mpl.rcParams['figure.subplot.left'] = .1
    mpl.rcParams['figure.subplot.right'] = .9
    mpl.rcParams['figure.subplot.top'] = .9
    mpl.rcParams['figure.subplot.bottom'] = .1
    mpl.rcParams['figure.subplot.hspace'] = 0.01
    mpl.rcParams['figure.subplot.wspace'] = 0.01  
    fig, ax = plt.subplots(figsize=(10, 6),
                       subplot_kw={'projection': ccrs.PlateCarree()} ) #ccrs.epsg(params['general']['crs']) })  # PlateCarree() == EPSG:4326

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
                gpd.GeoSeries([row_ctr.geometry]).set_crs(params['general']['crs']).plot(ax=ax, color=colors[irow], linewidth=0.1, zorder=2, alpha=0.7, markersize=1)
            else:
                gpd.GeoSeries([row_ctr.geometry]).set_crs(params['general']['crs']).to_crs(4326).plot(ax=ax, facecolor='none',edgecolor=colors[irow], cmap=cmap, linewidth=1, zorder=2, alpha=0.7)
            if flag_plot_hs:
                points = gpd.GeoSeries([row_hs.geometry]).explode(index_parts=False).set_crs(params['general']['crs'])
                if len(points)>0:
                    points.to_crs(4326).plot(ax=ax, color=colors[irow], alpha=0.5, markersize=40)
    
    for event in pastFireEvents:
        if flag_remove_singleHs: 
            if len(event.times) == 1: continue
        event.ctrs.set_crs(params['general']['crs']).to_crs(4326).plot(ax=ax, facecolor='none',edgecolor='k', alpha=0.2, linewidth=0.1, linestyle='--', zorder=1, markersize=1)

    ax.set_title(date_now)

    # Set extent if needed
    extentmm=params['general']['domain'].split(',')
    ax.set_extent([float(extentmm[i]) for i in [0,2,1,3]])
    ax.set_aspect(1)
    
    # Add gridlines with labels
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
   
    filename = "{:s}/Fig/fireEvent-{:s}-{:s}.png".format(params['event']['dir_data'],params['general']['domainName'],date_now.strftime("%Y-%m-%d_%H%M")) 
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    fig.savefig(filename,dpi=200)
    plt.close(fig)
    
    #plt.show()



############################
def run_fire_tracking(args):
    inputName = args.inputName
    sensorName = args.sensorName
    log_dir = args.log_dir
   
    #log_dir = sys.argv[2]
    #inputName = sys.argv[1]

    #init dir
    if inputName == '202504': 
        params = init(inputName,sensorName,log_dir) 
        start = datetime.strptime('2025-04-12_0000', '%Y-%m-%d_%H%M')
        end = datetime.strptime('2025-04-15_0000', '%Y-%m-%d_%H%M')
    
    elif inputName == 'AVEIRO': 
        params = init(inputName,sensorName,log_dir) 
        start = datetime.strptime('2024-09-15_0000', '%Y-%m-%d_%H%M')
        end = datetime.strptime('2024-09-20_2300', '%Y-%m-%d_%H%M')
    
    elif inputName == 'ofunato': 
        params = init(inputName,sensorName,log_dir) 
        start = datetime.strptime(params['general']['time_start'], '%Y-%m-%d_%H%M')
        end = datetime.strptime(params['general']['time_end'], '%Y-%m-%d_%H%M')
    
    elif 'SILEX' in inputName : 
        params = init(inputName,sensorName,log_dir) 
        if os.path.isfile(log_dir+'/timeControl.txt'): 
            with open(log_dir+'/timeControl.txt','r') as f:
                start = datetime.strptime(f.readline().strip(), '%Y-%m-%d_%H%M').replace(tzinfo=timezone.utc)
        else:
            start = datetime.strptime(params['event']['start_time'], '%Y-%m-%d_%H%M').replace(tzinfo=timezone.utc)
        
        #end = datetime.strptime('2025-06-27_0530', '%Y-%m-%d_%H%M')
        if params['general']['sensor'] == 'VIIRS':
            end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) 
        elif params['general']['sensor'] == 'FCI':
            end = (datetime.now(timezone.utc) - timedelta(minutes=20)) #20 minute of MTG latence.
            # Round down to the nearest xx:00 or xx:30
            end = end.replace(second=0, microsecond=0)
            if end.minute < 30:
                end = end.replace(minute=0)
            else:
                end = end.replace(minute=30) 
        
        end = end - timedelta(minutes=30) # so that the integration finish at time end calculated above
    else:
        print('missing inputName')
        sys.exit()
    
    #make sure all time are UTC
    start = start.replace(tzinfo=timezone.utc)
    end   = end.replace(tzinfo=timezone.utc) #- timedelta(hours=1)

    print('########times#########')
    print(start)
    print(end)
    print('######################')
   
    if start == end: 
        print('end == start: stop here')
        sys.exit()
    
    maskHS_da = None
    if 'mask_HS' in params['event']:
        if os.path.isfile(f"{src_dir}/../data_local/{params['event']['mask_HS']}"):
            maskHS_da = xr.open_dataarray(f"{src_dir}/../data_local/{params['event']['mask_HS']}").rio.write_crs("EPSG:4326", inplace=False)
    else:
        print(' ')
        print('## WARNING ####################')
        print('no hotspot mask was set in cong file')
        print('see src_extra/create_mask_fixFire to generate one')
        print('and set it in conf/event')
        print('## WARNING ####################')
        print(' ')

    # Loop hourly
    current = start
    end_time = None
    if params['general']['sensor'] == 'VIIRS':
        dt_minutes = 60.
    elif params['general']['sensor'] == 'FCI':
        dt_minutes = 30.
    
    while current <= end:
            
        end_time = current  
        #get last time processed
        #if os.path.isfile("{:s}/hotspots-{:s}.gpkg".format(params['event']['dir_data'],(current+timedelta(minutes=dt_minutes)).strftime("%Y-%m-%d_%H%M")) ):
        if os.path.isfile("{:s}/hotspots-{:s}.gpkg".format(params['event']['dir_data'],(current).strftime("%Y-%m-%d_%H%M")) ):
            print(current, end=' already done\n')
            current += timedelta(minutes=dt_minutes)
            continue 
        
        #track perimeter
        start_datetime = current.strftime('%Y-%m-%d_%H%M')
        #end_datetime   = (current + timedelta(hours=1)).strftime('%Y-%m-%d_%H%M')
        date_now, fireEvents, pastFireEvents = perimeter_tracking(params, start_datetime, maskHS_da, dt_minutes)#,end_datetime)
        
        #if date_now.hour in [0,3,6,9,12,15,18,21] and date_now.minute == 0:
        #    #ploting
        #    plot(params, date_now, fireEvents, pastFireEvents, flag_plot_hs=True, flag_remove_singleHs=True)

        #control hourly loop
        current += timedelta(minutes=dt_minutes)

    with open(log_dir+'/timeControl.txt','w') as f:
        f.write(end_time.strftime('%Y-%m-%d_%H%M'))


#############################
if __name__ == '__main__':
#############################
    
    '''
    SILEX
    domain: -10,35,20,46
    '''
    print('FET start!')
    importlib.reload(hstools)
    src_dir = os.path.dirname(os.path.abspath(__file__))
  
    parser = argparse.ArgumentParser(description="fireEventTracking")
    parser.add_argument("--inputName", type=str, help="name of the configuration input", )
    parser.add_argument("--sensorName", type=str, help="name of the sensor, VIIRS or FCI", )
    parser.add_argument("--log_dir", type=str, help="Directory for logs", default='/mnt/dataEstrella2/SILEX/VIIRS-HotSpot/FireEvents/log/')

    args = parser.parse_args()

    run_fire_tracking(args)

    print('FET done!')
    '''
    import cProfile
    import pstats
    cProfile.run('run_fire_tracking(args)', 'profile_output')

    p = pstats.Stats('profile_output')
    p.strip_dirs().sort_stats('cumtime').print_stats(20)
    '''
