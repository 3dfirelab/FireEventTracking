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
import matplotlib.dates as mdates
import pickle

warnings.filterwarnings("error", category=pd.errors.SettingWithCopyWarning)

#home brewed
import hstools
import fireEvent
import os
import subprocess
import fireEventTracking as fet

###############################################
def plot(params, year, week, fireEvents, pastFireEvents, flag_plot_hs=True, flag_remove_singleHs=False):
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
                #gpd.GeoSeries([row_ctr.geometry]).set_crs(params['general']['crs']).plot(ax=ax, color=colors[irow], linewidth=0.1, zorder=2, alpha=0.7, markersize=1)
                gpd.GeoSeries([row_ctr.geometry]).set_crs(params['general']['crs']).to_crs(4326).plot(ax=ax, color='k', linewidth=0.1, zorder=2, alpha=0.7, markersize=1)
                print('point')
            else:
                gpd.GeoSeries([row_ctr.geometry]).set_crs(params['general']['crs']).to_crs(4326).plot(ax=ax, facecolor='none',edgecolor=colors[irow], cmap=cmap, linewidth=1, zorder=2, alpha=0.7)
            
            if flag_plot_hs:
                points = gpd.GeoSeries([row_hs.geometry]).explode(index_parts=False).set_crs(params['general']['crs'])
                if len(points)>0:
                    points.to_crs(4326).plot(ax=ax, color=colors[irow], alpha=0.5, markersize=40)
    
    for event in pastFireEvents:
        for (irow,row_ctr), (_,row_hs) in zip(event.ctrs.iterrows(),event.hspots.iterrows()):
            #if flag_remove_singleHs: 
            #    if len(event.times) == 1: continue
     
            if row_ctr.geometry.geom_type != 'Polygon':
                gpd.GeoSeries([row_ctr.geometry]).set_crs(params['general']['crs']).to_crs(4326).plot(ax=ax, color='r', alpha=0.9, linewidth=0.1, zorder=1, markersize=1)
            else: 
                gpd.GeoSeries([row_ctr.geometry]).set_crs(params['general']['crs']).to_crs(4326).plot(ax=ax, facecolor='none',edgecolor='r', alpha=0.9, linewidth=0.1 , zorder=1, markersize=1)

    ax.set_title(f'{year} - week{week}' )

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
   
    filename = "{:s}/Stats/fireEvent-{:s}-{:d}-{:02d}.png".format(params['event']['dir_data'],params['general']['domainName'],year,week) 
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    fig.savefig(filename,dpi=400)
    plt.close(fig)
    
    #plt.show()


############################
def run_fire_stats(args):
    inputName = args.inputName
    sensorName = args.sensorName
    #log_dir = args.log_dir
   
    #log_dir = sys.argv[2]
    #inputName = sys.argv[1]

    #init dir
    if inputName == '202504': 
        params = fet.init(inputName,sensorName) 
        start = datetime.strptime('2025-04-12_0000', '%Y-%m-%d_%H%M')
        end = datetime.strptime('2025-04-15_0000', '%Y-%m-%d_%H%M')
    
    elif inputName == 'AVEIRO': 
        params = fet.init(inputName,sensorName) 
        start = datetime.strptime('2024-09-15_0000', '%Y-%m-%d_%H%M')
        end = datetime.strptime('2024-09-20_2300', '%Y-%m-%d_%H%M')
    
    elif inputName == 'ofunato': 
        params = fet.init(inputName,sensorName) 
        start = datetime.strptime(params['general']['time_start'], '%Y-%m-%d_%H%M')
        end = datetime.strptime(params['general']['time_end'], '%Y-%m-%d_%H%M')
    
    elif inputName == 'RIBAUTE': 
        params = fet.init(inputName,sensorName) 
        start = datetime.strptime(params['event']['start_time'], '%Y-%m-%d_%H%M')
        end = datetime.strptime(params['event']['end_time'], '%Y-%m-%d_%H%M')
    
    elif ('MED' in inputName ): 
        params = fet.init(inputName,sensorName) 
        if False: 
            print('*************************************')
            print('************** set start and end time')
            print('*************************************')
            start = datetime.strptime('2025-02-01_0000', '%Y-%m-%d_%H%M' ) # params['general']['time_start'], '%Y-%m-%d_%H%M')
            end = datetime.strptime(  '2025-03-31_2350', '%Y-%m-%d_%H%M') #params['general']['time_end'], '%Y-%m-%d_%H%M')
            print(start)
            print(end)
            
        else:

            #if os.path.isfile(log_dir+'/timeControl.txt'): 
            #    with open(log_dir+'/timeControl.txt','r') as f:
            #        start = datetime.strptime(f.readline().strip(), '%Y-%m-%d_%H%M').replace(tzinfo=timezone.utc)
            #else:
            start = datetime.strptime(params['event']['start_time'], '%Y-%m-%d_%H%M').replace(tzinfo=timezone.utc)
            end   = datetime.strptime(params['event']['end_time_hard'], '%Y-%m-%d_%H%M').replace(tzinfo=timezone.utc)

    elif ('SILEX' in inputName) or ('PORTUGAL' in inputName)  or ('MED' in inputName ): 
        params = fet.init(inputName,sensorName) 
        if os.path.isfile(log_dir+'/timeControl.txt'): 
            with open(log_dir+'/timeControl.txt','r') as f:
                start = datetime.strptime(f.readline().strip(), '%Y-%m-%d_%H%M').replace(tzinfo=timezone.utc)
        else:
            start = datetime.strptime(params['event']['start_time'], '%Y-%m-%d_%H%M').replace(tzinfo=timezone.utc)
        
        #end = datetime.strptime('2025-06-27_0530', '%Y-%m-%d_%H%M')
        if params['general']['sensor'] == 'VIIRS':
            end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) 
        elif params['general']['sensor'] == 'FCI':
            if 'end_time' in params['event']: 
                end = datetime.strptime(params['event']['end_time'], '%Y-%m-%d_%H%M').replace(tzinfo=timezone.utc)
            else:
                end = (datetime.now(timezone.utc) - timedelta(minutes=20)) #20 minute of MTG latence.
                # Round down to the nearest xx:00 or xx:30
                end = end.replace(second=0, microsecond=0)
                if end.minute < 30:
                    end = end.replace(minute=0)
                else:
                    end = end.replace(minute=30) 
        
        #end = end - timedelta(minutes=30) # so that the integration finish at time end calculated above
        #print('**************   hard set end time')
        #end = datetime.strptime('2025-08-06_1000', '%Y-%m-%d_%H%M')
    
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

    root = Path(params['event']['dir_data'])
    os.makedirs(params['event']['dir_data']+'/Stats/', exist_ok=True)
    os.makedirs(f"{params['event']['dir_data']}/FRP-FROS/", exist_ok=True)
    os.makedirs(f"{params['event']['dir_data']}/Stats/GeoJson", exist_ok=True)
    # regex to extract datetime from directory name
    pattern = re.compile(r"Pickles_active_(\d{4}-\d{2}-\d{2}_\d{4})")

    records = []

    for dir_ in root.glob("Pickles_active_*"):
        m = pattern.search(dir_.name)
        if not m:
            continue
        dt_str = m.group(1)  # '2025-03-25_1630'
        dt = pd.to_datetime(dt_str, format="%Y-%m-%d_%H%M")
        
        for pkl in dir_.glob("*.pkl"):
            fire_id = pkl.stem  # basename without .pkl
            records.append((fire_id, pkl, dt))
    
    df = pd.DataFrame(records, columns=["fire_id", "file", "datetime"])
    df['fire_id'] = df['fire_id'].astype(int)
    
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize("UTC")
    df = df[ (df["datetime"]>=start) & (df["datetime"]<=end) ]

    # keep only the last file per fire_id (latest datetime)
    df_sorted = df.sort_values(by=["fire_id", "datetime"])
    df_latest = df_sorted.groupby("fire_id").tail(1).reset_index(drop=True)

    # final dataframe with filename and date
    df_final = df_latest[["file", "datetime","fire_id"]].copy()


    # assign ISO week + year (avoid collisions across years)
    df_final["year"] = df_final["datetime"].dt.isocalendar().year
    df_final["week"] = df_final["datetime"].dt.isocalendar().week
    df_final["month"] = df_final["datetime"].dt.month
   
    
    # group by ISO year-week
    grouped = df_final.groupby(["year", "week",])

    #to get fire id of fire that merged
    #print('load merging fie info: ')
    #sys.stdout.flush()
    #fire_merged = {}
    #for ie, event_file in enumerate(df_final.file):
    #    print ( f'{ie*100./ df_final.shape[0]:.2f} %', end='\r')
    #    event = fireEvent.load_fireEvent(event_file)
    #    if len(event.id_fire_event_dad) == 0: 
    #        continue
    #    else:
    #        for id_ in event.id_fire_event_dad:
    #            fire_merged[id_]=event.id_fire_event
    #print('done               ')

    list_pb_in_merged_fire = []
    
    gdfs = []

    data_per_week = []
    data_per_week_all = []
    #gdf_events_all = None
    for (year, week,), df_week in grouped:
        print(f"Processing YEAR={year}, WEEK={week}, N={len(df_week)}")

        # inner loop: iterate pkl files within the week
        weekEvents = []
        weekEvents_files = []
        weekFRP = 0
        weekNbreFire = 0 
        weekArea_arr = []
        weekDuration_arr = []
        for _, row in df_week.iterrows():
            fire_id = row["fire_id"]
            event_file = row["file"]
            dt = row["datetime"]

            event = fireEvent.load_fireEvent(event_file)
            weekEvents.append( event ) 
            weekEvents_files.append(event_file)
         
            #saved FRP and FROS time series for each event in FRP_FROS dir
            if len(event.fros) == len(event.frps) : 
                    dfts = pd.DataFrame({'timestamp':event.times, 'frp':event.frps, 'fros':event.fros})
                    dfts["timestamp"] = pd.to_datetime(dfts["timestamp"], utc=True)
                    dfts[["timestamp", "frp", "fros"]].to_json(
                                              f"{params['event']['dir_data']}/FRP-FROS/{fire_id:09d}.json",
                                                orient="records",
                                                date_format="iso"
                                              )
            else:
                print(f"{fire_id:09d} len fros != frp {len(event.fros)}  {len(event.frps)}")

            if False: #fire_id == 141: 
                fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
                ax=axes[0]
                ax.plot(event.times, event.frps)
                ax.set_ylabel('FRP (MW)')
                ax=axes[1]
                fros = np.array(event.fros, dtype=float)
                fros[fros == -999] = np.nan
                ax.plot(event.times, fros)
                ax.set_ylabel('FROS (m/s)')
                ax=axes[2]
                area = np.array(event.ctrs.geometry.area, dtype=float) * 1.e-4
                ax.plot(event.times, area)
                ax.set_ylabel('burning Aera (ha)')
               
                fig, axes = plt.subplots(1, 1, figsize=(6, 6), sharex=True)
                gdf_effis = gpd.read_file('/data/paugam/FIRES/effis_modis_ba_ribaute.geojson')
                gdf_effis = gdf_effis.to_crs(event.ctrs.crs)
                gdf_ = event.ctrs.copy()
                gdf_["fros"] = np.array(fros, dtype=float)
                gdf_["time"] = np.array(event.times)
                gdf_ = gdf_.sort_values("time", ascending=False)
                ax = axes
                gdf_.plot(column='fros',cmap="viridis", legend=True, ax=ax, edgecolor="black", linewidth=0.2)
                gdf_effis.plot(ax=ax, facecolor='none',edgecolor='r', alpha=0.7)
                ax.set_title("FROS")
                plt.show()
                pdb.set_trace()

            weekFRP += np.array(event.frps).sum()
            weekNbreFire += 1
            
            geom_final = event.ctrs.iloc[-1]
            if isinstance(geom_final.geometry, (Polygon, MultiPolygon)):
                area_event = event.ctrs.iloc[-1].geometry.area * 1.e-4 
            else:
                area_event = 0
            weekArea_arr.append(area_event)
            
            if len(event.times)>=2: 
                timeDuration_event = (event.times[-1]-event.times[0]).total_seconds() / 3600.
            else:
                timeDuration_event = 0
            weekDuration_arr.append(timeDuration_event)

        data_per_week.append([year, week, weekFRP, weekNbreFire, np.array(weekArea_arr).mean(), np.array(weekArea_arr).std(), np.array(weekDuration_arr).mean(), np.array(weekDuration_arr).std() ] )
        data_per_week_all.append([weekArea_arr, weekDuration_arr])

        plot(params, year, week, [], weekEvents, flag_plot_hs=False, flag_remove_singleHs=True)
   
        
        for event,event_file in zip(weekEvents,weekEvents_files):
            
            frp_ = np.array(event.frps)              # MW
            t_ = pd.to_datetime(event.times)
            t_sec = (t_ - t_[0]).total_seconds().values
            fre_ = np.trapezoid(frp_, t_sec)
            
            t_ = pd.to_datetime(event.times)
         
            time_merge = event.time_merging[0] if (len(event.time_merging)==1) else None
            
            if len(event.time_include) == 0: 
                times_include = None
            else:
                times_include = ';'.join( [xx.strftime("%Y-%m-%d_%H:%M:%S") for xx in event.time_include] )

            gdf_ = gpd.GeoDataFrame(
                    {
                        "center_igni":event.centers[0], 
                        "time_start": t_.floor("30 min")[0],
                        "time_end": t_.floor("30 min")[-1],
                        "time_merged": time_merge,
                        "fre": fre_,
                        "fire_event_id": event.id_fire_event,
                        "fire_name": event.fire_name,
                        "file": event_file,
                        "dad_event": ';'.join(map(str, event.id_fire_event_dad)),
                        "son_event": ';'.join(map(str, event.id_fire_event_son)),
                        "time_include": times_include,
                    },
                    geometry=[event.ctrs['geometry'].iloc[-1]],
                    crs=event.ctrs.crs, 
                    index=[0],
                )
            #save indiviual geojson per event
            #event = fireEvent.load_fireEvent(event_file)
            gdf__ = gpd.GeoDataFrame(
                                        {
                                            "time": t_, 
                                            "time_floor": t_.floor("30 min"), 
                                            "frp": frp_, 
                                            "geometry": event.ctrs.geometry.values,
                                        },
                                        crs=event.ctrs.crs,   # preserve original CRS
                                   )
            filename_fire_feature = f"{params['event']['dir_data']}/Stats/GeoJson/gdf_{event.id_fire_event}.geojson"
            gdf__.to_file(filename_fire_feature, driver="GeoJSON")

            gdfs.append(gdf_)

        #    if gdf_events_all is None:
        #        gdf_events_all = gdf_
        #    else:
        #        gdf_events_all = pd.concat([gdf_, gdf_events_all])

    # single concat → stable dtype inference
    gdf_events_all = gpd.GeoDataFrame(
        pd.concat(gdfs, ignore_index=True),
        crs=gdfs[0].crs if gdfs else None )

    df_weekly = pd.DataFrame(
                                data_per_week,
                                columns=[
                                    "year",
                                    "week",
                                    "total_frp",
                                    "event_count",
                                    "mean_area",
                                    "std_area",
                                    "mean_duration_h",
                                    "std_duration_h",
                                ]
                            )
    
    df_weekly.to_csv("{:s}/Stats/{:s}-weekly.csv".format(params['event']['dir_data'],params['general']['domainName']))
    with open("{:s}/Stats/{:s}-area_duration_weekly_allData.pkl".format(params['event']['dir_data'],params['general']['domainName']), "wb") as f:
        pickle.dump(data_per_week_all, f) 
    
    gdf_events_all.to_file("{:s}/Stats/{:s}-gdf_{:s}_{:s}.geojson".format(
                            params['event']['dir_data'],params['general']['domainName'], start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")), driver="GeoJSON")

    if len(list_pb_in_merged_fire) > 0: 
        print(f'found {len(list_pb_in_merged_fire)} with tagged fire son that does not contains stoped fire dad')
        print(list_pb_in_merged_fire)

    return df_weekly, gdf_events_all

#############################
if __name__ == '__main__':
#############################
    
    '''
    SILEX
    domain: -10,35,20,46
    '''
    #print('FET start!')
    importlib.reload(hstools)
    importlib.reload(fireEvent)
    src_dir = os.path.dirname(os.path.abspath(__file__))
  
    parser = argparse.ArgumentParser(description="fireEventTracking")
    parser.add_argument("--inputName", type=str, help="name of the configuration input", )
    parser.add_argument("--sensorName", type=str, help="name of the sensor, VIIRS or FCI", )
    #parser.add_argument("--log_dir", type=str, help="Directory for logs", default='/mnt/dataEstrella2/SILEX/VIIRS-HotSpot/FireEvents/log/')

    args = parser.parse_args()

    df_weekly, gdf_events_all = run_fire_stats(args)

