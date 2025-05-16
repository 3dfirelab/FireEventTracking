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
    

    df = None
    for satname in params['hs']['sats']:
        hsfiles = glob.glob('{:s}/{:s}/*{:s}-{:s}.csv'.format(params['hs']['dir_data'],satname,str(day),hour))
        if len(hsfiles)==1: 
            try:
                df_ = pd.read_csv(hsfiles[0], delimiter=',', header=0)
            except EmptyDataError: 
                continue
            if len(df_) == 0 : continue
            if df is None: 
                df = df_
            else:
                df = pd.concat([df,df_])
        
        elif len(hsfiles)>1:
            for file_ in sorted(hsfiles):
                if '0000' in file_: continue # if here we want to load data from last day, 0000 is the data from the day d-2
                df__ = pd.read_csv(file_, delimiter=',', header=0)
                if '2025-03-19' in df__['acq_date'].astype(str).values: pdb.set_trace()
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
   
    gdf = gdf.dropna(subset=["acq_time"])

    # Combine and convert to datetime
    gdf["timestamp"] = pd.to_datetime(gdf["acq_date"] + " " + gdf["acq_time"].astype(int).astype(str).str.zfill(4), format="%Y-%m-%d %H%M")

    # Set the coordinate reference system (CRS), assuming WGS84 (EPSG:4326)
    gdf.set_crs(epsg=4326, inplace=True)

    return gdf.to_crs(params['general']['crs']).reset_index(drop=True)


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

