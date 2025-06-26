import numpy as np 
import pandas as pd
import glob
import pdb 
import yaml 
from shapely.geometry import Point
import geopandas as gpd
import matplotlib.pyplot as plt 
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta
from shapely.geometry import box

from pandas.errors import EmptyDataError

#############################
def load_config(path: str):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config


#############################
def load_hs4oneday(day,sat,params):
    '''
    return all hs from the given day and sat
    '''
    hsfiles = glob.glob('{:s}/{:s}/*{:s}*.csv'.format(params['hs']['dir_data'],sat,str(day)))
    df = None
    for file_ in hsfiles:
        df_ = pd.read_csv(file_, delimiter=',', header=0)
        if len(df_) == 0 : continue
        if df is None: 
            df = df_
        else:
            df = pd.concat([df,df_])
   
    if df is None: 
        columns = [
                    "latitude", "longitude", "bright_ti4", "scan", "track", "acq_date",
                    "acq_time", "satellite", "instrument", "confidence", "version",
                    "bright_ti5", "frp", "daynight", "geometry", "timestamp",
                   ]
        # Create the empty DataFrame
        return  pd.DataFrame(columns=columns)

    # Create geometry column
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=geometry)
    
    # Combine and convert to datetime
    gdf["timestamp"] = pd.to_datetime(gdf["acq_date"] + " " + gdf["acq_time"].astype(str).str.zfill(4), format="%Y-%m-%d %H%M")

    # Set the coordinate reference system (CRS), assuming WGS84 (EPSG:4326)
    gdf.set_crs(epsg=4326, inplace=True)

    return gdf.to_crs(params['general']['crs'])

#############################
def load_hs4lastObsAllSat(day,hour,params):
    '''
    return all hs from the given day/hour for all sat
    '''
    if params['general']['sensor'] == 'VIIRS':
         satnames = ['VIIRS_NOAA20_NRT', 'VIIRS_NOAA21_NRT', 'VIIRS_SNPP_NRT']
    elif params['general']['sensor'] == 'FCI':
         satnames = ['']
    

    df = None
    for satname in satnames:
       
        if params['general']['sensor'] == 'VIIRS':
            hsfiles = glob.glob('{:s}/{:s}/*{:s}-{:s}.csv'.format(params['hs']['dir_data'],satname,str(day),hour))
        elif params['general']['sensor'] == 'FCI':
            dt = datetime.strptime(day + hour, "%Y-%m-%d%H%M")
            ## Subtract 1 hour
            #dt_minus_1h = dt - timedelta(hours=1)
            # Extract new day and hour
            #new_day = dt_minus_1h.strftime("%Y-%m-%d")
            #new_hour = dt_minus_1h.strftime("%H%M")
            new_day = dt.strftime("%Y-%m-%d")
            new_hour = dt.strftime("%H%M")
            try: 
                hsfiles = sorted(glob.glob('{:s}/{:s}/*{:s}{:s}*.csv'.format(params['hs']['dir_data'],satname,str(new_day).replace('-',''),new_hour)))
            except: 
                pdb.set_trace()
        
        if len(hsfiles)==1: 
            try:
                df_ = pd.read_csv(hsfiles[0], delimiter=',', header=0)
                df_ = df_.dropna()
            except EmptyDataError: 
                continue
            if len(df_) == 0 : continue
            if df is None: 
                df = df_
            else:
                df = pd.concat([df,df_])
        
        elif len(hsfiles)>1:
            for file_ in sorted(hsfiles):
                if params['general']['sensor'] == 'VIIRS':
                    if '0000' in file_: continue # if here we want to load data from last day, 0000 is the data from the day d-2
                df__ = pd.read_csv(file_, delimiter=',', header=0)
                if len(df__)==0: continue
                if df is None:
                    df_ = pd.concat([df__,df,df]).drop_duplicates(keep=False)
                else: 
                    df_ = df__
                if len(df_) == 0 : continue
                if df is None: 
                    df = df_
                else:
                    df = pd.concat([df,df_])

        elif len(hsfiles)==0:
            continue
   
    try:
        if params['general']['sensor'] == 'FCI':
            df.rename(columns={'LONGITUDE': 'longitude'}, inplace=True)
            df.rename(columns={'LATITUDE': 'latitude'}, inplace=True)
            df.rename(columns={'FRP': 'frp'}, inplace=True)
    except: 
        pdb.set_trace()

    if df is None:
        if params['general']['sensor'] == 'VIIRS':
            columns = [
                        "latitude", "longitude", "bright_ti4", "scan", "track", "acq_date",
                        "acq_time", "satellite", "instrument", "confidence", "version",
                        "bright_ti5", "frp", "daynight", "geometry", "timestamp",
                       ]
        elif params['general']['sensor'] == 'FCI':
                    columns = ['BW_SIZE','BW_NUMPIX','BW_BT_MIR','BW_BTD','RAD_BCK','STD_BCK','FIRE_CONFIDENCE','BT_MIR','BT_TIR',
                               'RAD_PIX','PIXEL_VZA','PIXEL_SZA','PIXEL_SIZE','Longitude','latitude','ACQTIME','ABS_line','ABS_samp',
                               'frp','ERR_FRP_COEFF','ERR_ATM_TRANS','ERR_RADIOMETRIC','ERR_BACKGROUND','ERR_VERT_COMP','FRP_UNCERTAINTY','PIXEL_ATM_TRANS',
                               "geometry", "timestamp",
                        ]

        # Create the empty DataFrame
        return  pd.DataFrame(columns=columns)


    # Create geometry column
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=geometry)
    gdf.set_crs(epsg=4326, inplace=True)
    
    if params['general']['sensor'] == 'FCI':
        #apply clip to domain
        bbox_geom = box(*[float(xx) for xx in params["general"]["domain"].split(',')])
        bbox_gdf  = gpd.GeoDataFrame(geometry=[bbox_geom], crs="4326")
        gdf = gpd.clip(gdf, bbox_gdf).reset_index(drop=True)

    
    # Combine and convert to datetime
    if params['general']['sensor'] == 'VIIRS':
        gdf = gdf.dropna(subset=["acq_time"])
        gdf["timestamp"] = pd.to_datetime(gdf["acq_date"] + " " + gdf["acq_time"].astype(int).astype(str).str.zfill(4), format="%Y-%m-%d %H%M")
    elif params['general']['sensor'] == 'FCI':
        gdf = gdf.dropna(subset=["ACQTIME"])
        dt_ref = datetime.strptime(day + hour, "%Y-%m-%d%H%M")
        #new_day = dt.strftime("%Y-%m-%d")
        #new_hour = dt.strftime("%H%M")
        
        #dt = datetime.strptime(day + hour, "%Y-%m-%d%H%M")
        ## Subtract 1 hour
        #dt_minus_1h = dt - timedelta(hours=1)
        ## Extract new day and hour
        #new_day = dt_minus_1h.strftime("%Y-%m-%d")
        #new_hour = dt_minus_1h.strftime("%H%M")
        #dt_ref = datetime.strptime(f"{new_day} {new_hour}", "%Y-%m-%d %H%M")
        
        gdf["timestamp"] = gdf['ACQTIME'].apply(lambda x: convert_acqtime(x, dt_ref))
        
        #try: 
        #    gdf["timestamp"] = dt_ref + pd.to_timedelta(gdf['ACQTIME'].values, unit='s')
        #except: 
        #    pdb.set_trace()
        #    gdf["timestamp"] = pd.to_datetime(gdf['ACQTIME'].astype(str), format='%Y%m%d%H%M%S')

    return gdf.to_crs(params['general']['crs']).reset_index(drop=True)


#########################
def convert_acqtime(val, dt_ref):
    try:
        if val < 600:
            return dt_ref + timedelta(seconds=val)
        else:
            return datetime.strptime(str(val), '%Y%m%d%H%M%S')
    except:
        return pd.NaT  # Handle unexpected values safely


#########################
def plot_hs(gdf,params):

    # Create a figure with Cartopy
    fig, ax = plt.subplots(figsize=(10, 6),
                       subplot_kw={'projection': ccrs.epsg(params['general']['crs']) })  # PlateCarree() == EPSG:4326

    # Add basic map features
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

    # Plot the GeoDataFrame
    gdf.plot(ax=ax, transform=ccrs.epsg(params['general']['crs']), column='global_fire_event', facecolor='none', cmap='jet', alpha=0.1)

    # Set extent if needed
    extent=params['general']['domain'].split(',')
    ax.set_extent([extent[i] for i in [0,2,1,3]])

    # Add gridlines with labels
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    return ax

