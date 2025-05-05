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
warnings.filterwarnings("error", category=pd.errors.SettingWithCopyWarning)

#home brewed
import hstools
import fireEvent

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
def init(config_name):
    script_dir = Path(__file__).resolve().parent
    params = hstools.load_config(str(script_dir)+f'/../config/config-{config_name}.yaml')
    
    if socket.gethostname() == 'moritz': 
        params['hs']['dir_data'] = params['hs']['dir_data'].replace('/mnt/data3/','/mnt/dataEstrella2/')
        params['event']['dir_data'] = params['event']['dir_data'].replace('/mnt/data3/','/mnt/dataEstrella2/')

    os.makedirs(params['event']['dir_data'],exist_ok=True)
    

    return params
  

####################################################
def perimeter_tracking(params, start_datetime,end_datetime, flag_restart=False):
    
    start_datetime = datetime.strptime(f'{start_datetime}', '%Y-%m-%d_%H%M')
    end_datetime   = datetime.strptime(f'{end_datetime}', '%Y-%m-%d_%H%M')

    date_now = start_datetime
    
    #load fire event
    fireEvents = []
    pastFireEvents = []
    hsgdf_all_raw = None
  
    #load last active fire saved as well as the past event. past event older than 7 days are not loaded
    if os.path.isfile("{:s}/hotspots-{:s}.gpkg".format(params['event']['dir_data'],start_datetime.strftime("%Y-%m-%d_%H%M")) ):
        #activate loading existing data
        hsgdf_all_raw = gpd.read_file("{:s}/hotspots-{:s}.gpkg".format(params['event']['dir_data'],start_datetime.strftime("%Y-%m-%d_%H%M")) )

        fireEvents_files = sorted(glob.glob( params['event']['dir_data']+'/Pickles_{:s}_{:s}/*.pkl'.format('active',start_datetime.strftime("%Y-%m-%d_%H%M"))))
        ii = 0
        for id_, event_file in enumerate(fireEvents_files):
            event = fireEvent.load_fireEvent(event_file)
            while ii < event.id_fire_event:
                fireEvents.append(None)
                ii += 1
            fireEvents.append( fireEvent.load_fireEvent(event_file) ) 
            ii += 1
        fireEvent.Event._id_counter = fireEvents[-1].id_fire_event +1

        pastFireEvents_files = sorted(glob.glob( params['event']['dir_data']+'/Pickles_{:s}/*.pkl'.format('past',)))
        for id_, event_file in enumerate(pastFireEvents_files):
            try: 
                end_time_ = datetime.strptime( '_'.join(event_file.split('.')[0].split('_')[-2:]), "%Y-%m-%d_%H%M")
                if end_time_ > start_datetime - timedelta(days=7):
                    pastFireEvents.append( fireEvent.load_fireEvent(event_file) ) 
            except: 
                pdb.set_trace()
        
        #date_now = date_now + timedelta(hours=1)    


    idate = 0
    while date_now<end_datetime:
        print(date_now, end='')
        

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
            print(' skip')
            date_now = date_now + timedelta(hours=1)    # subtract one day
            idate+=1
            continue
       
        
        if hsgdf_all_raw is None:
            hsgdf_all_raw = hsgdf.copy()
        else: 
            duplicates = pd.merge(hsgdf_all_raw.assign(version=hsgdf_all_raw['version'].astype(str)), hsgdf.assign(version=hsgdf_all_raw['version'].astype(str)), how='inner')
            if len(duplicates)>0: 
                print('')
                print('find duplicate hotspot in new entry')
                pdb.set_trace()
            hsgdf.index = range(hsgdf_all_raw.index.max() + 1, hsgdf_all_raw.index.max() + 1 + len(hsgdf))
            try:
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
        
        fireCluster = fireCluster.to_crs(params['general']['crs'])
    

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
            print(' create first fireEvents')
            for (_,cluster), (_,ctr) in zip(fireCluster.iterrows(),fireCluster_ctr.iterrows()):
                event = fireEvent.Event(cluster,ctr,fireCluster.crs,hsgdf_all_raw) 
                fireEvents.append(event)
        else: 
            print(' append to fireEvents')
            gdf_activeEvent = create_gdf_fireEvents(fireEvents)
            
            #here we go over each cluster and assign it to an existing event if its center is inside an active fire event. if not, this is a new one
            for (_,cluster), (_,ctr) in zip(fireCluster.iterrows(),fireCluster_ctr.iterrows()):
               
                if cluster.end_time < date_now: 
                    continue

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
                            #pastFireEvents.append(element)

                        gdf_activeEvent = create_gdf_fireEvents(fireEvents)
                        #flag_found_matchingEvent = True
                        continue

                    '''
                    #for point in active fire event
                    #-------------
                    gdf_points = gdf_activeEvent[gdf_activeEvent.geometry.type == 'Point'].copy()
                    # Check which polygons contain the point
                    
                    gdf_polygons_ =  gpd.GeoSeries([ctr.geometry],crs=gdf_points.crs)
                    active_points_inside_cluster = gdf_points[gdf_points.geometry.apply(lambda point: gdf_polygons_.iloc[0].contains(point))].copy()
                    
                    #if 124 in active_points_inside_cluster.index: pdb.set_trace()

                    # Get the polygons that contain the point
                    if len(active_points_inside_cluster)==1:
                        idx_event = active_points_inside_cluster.index[0]
                        fireEvents[idx_event].add(cluster,ctr,fireCluster.crs,hsgdf_all_raw)
                        #gdf_activeEvent = create_gdf_fireEvents(fireEvents)
                        #flag_found_matchingEvent = True
                        continue
                    elif len(active_points_inside_cluster)>1:
                        active_points_inside_cluster['distancetopoly'] = active_points_inside_cluster.geometry.distance(ctr.geometry.centroid).copy()
                        idx_event = active_points_inside_cluster['distancetopoly'].idxmin()
                        fireEvents[idx_event].add(cluster,ctr,fireCluster.crs,hsgdf_all_raw)
                        
                        #merge with bigger
                        other_indices = active_points_inside_cluster.index.difference([idx_event]).tolist()
                        for index_ in other_indices:
                            fireEvents[idx_event].merge(fireEvents[index_])
                        
                        fireEvents[idx_event].mergeWith(other_indices)
                        
                        #set merged event to past event
                        for index_ in other_indices:
                            #print('--', index_)
                            element = fireEvents[index_]
                            fireEvents[index_] = None
                            #pastFireEvents.append(element)

                        #gdf_activeEvent = create_gdf_fireEvents(fireEvents)
                        #flag_found_matchingEvent = True
                        continue

                    '''

                    #gdf_points['distance_to_center'] = gdf_points.geometry.distance(ctr.geometry)
                    #ros_max = 0.1
                    #nearby_points = gdf_points[gdf_points['distance_to_center'] < ros_max * (cluster.end_time-gdf_points.time).dt.total_seconds() ]
                    #nearby_points = gdf_points[gdf_points['distance_to_center'] < 5.e3 ]
                    #if len(nearby_points) == 1: 
                    #    idx_event = nearby_points.distance_to_center.idxmin()
                    #    fireEvents[idx_event].add(cluster,ctr,fireCluster.crs, hsgdf_all_raw)
                    #    continue
                    
                    #for line in active fire event, merge with line new cluster
                    #-------------
                    #if ctr.geometry.geom_type == 'LineString':
                    #    gdf_lines = gdf_activeEvent[gdf_activeEvent.geometry.type == 'LineString'].copy()
                    #    gdf_lines['distance_to_lines'] = gdf_lines.geometry.distance(ctr.geometry)
                    #    nearby_lines = gdf_lines[gdf_lines['distance_to_lines'] < 5.e3 ]
                    #    if len(nearby_lines) == 1:
                    #        idx_event = nearby_lines.distance_to_lines.idxmin()
                    #        fireEvents[idx_event].add(cluster,ctr,fireCluster.crs, hsgdf_all_raw)
                    #        continue
                
                #if flag_found_matchingEvent: 
                #    continue

                if (ctr.geometry.geom_type == 'Point') | (ctr.geometry.geom_type == 'LineString'):

                    '''
                    #for polygon in active fire event
                    #-------------
                    gdf_polygons = gdf_activeEvent[(gdf_activeEvent.geometry.type == 'Polygon')].copy()
                    # Check which polygons contain the point
                    contains_mask = gdf_polygons.contains(ctr.geometry)
                    # Get the polygons that contain the point
                    matching_polygons = gdf_polygons[contains_mask]
                    if len(matching_polygons) == 1: 
                        pdb.set_trace()
                        idx_event = matching_polygons.iloc[0].index
                        fireEvents[idx_event].add(cluster,ctr,fireCluster.crs, hsgdf_all_raw)
                        gdf_activeEvent = create_gdf_fireEvents(fireEvents)
                        continue
                    
                    #for point in active fire event
                    #-------------
                    gdf_points = gdf_activeEvent[gdf_activeEvent.geometry.type == 'Point'].copy()
                    gdf_points['distance_to_center'] = gdf_points.geometry.distance(ctr.geometry)
                    ros_max = 0.1
                    #nearby_points = gdf_points[gdf_points['distance_to_center'] < ros_max * (cluster.end_time-gdf_points.time).dt.total_seconds() ]
                    nearby_points = gdf_points[gdf_points['distance_to_center'] < 5.e3 ]
                    if len(nearby_points) == 1: 
                        pdb.set_trace()
                        idx_event = nearby_points.distance_to_center.idxmin()
                        fireEvents[idx_event].add(cluster,ctr,fireCluster.crs, hsgdf_all_raw)
                        gdf_activeEvent = create_gdf_fireEvents(fireEvents)
                        continue
                    '''
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
                            #pastFireEvents.append(element)

                        gdf_activeEvent = create_gdf_fireEvents(fireEvents)
                        #flag_found_matchingEvent = True
                        continue

                #if we are here, we have a new event
                #if 19604 in cluster.indices_hs: pdb.set_trace()
                #if 14634 in cluster.indices_hs: pdb.set_trace()
                #if cluster.frp in gdf_activeEvent['frp'].values: pdb.set_trace()
                new_event = fireEvent.Event(cluster,ctr,fireCluster.crs, hsgdf_all_raw) 
                fireEvents.append(new_event)
            

        #remove fireEvent that were updated more than two day ago. 
        for id_, event in enumerate(fireEvents):
            if event is None: continue
            if event.times[-1] < (date_now - timedelta(days=2)):
                element = fireEvents[id_]
                fireEvents[id_] = None
                pastFireEvents.append(element)
        gdf_activeEvent = create_gdf_fireEvents(fireEvents)
        
        #remove old hotspot older than 7 days
        hsgdf_all_raw = hsgdf_all_raw[hsgdf_all_raw['timestamp']>=date_now - timedelta(days=7)]

        #end temporal loop
        date_now = date_now + timedelta(hours=1)    # subtract one day
        idate+=1

    #save 
    #clean fire event dir
    #if os.path.isdir(params['event']['dir_data']+'Pickles_active/'):
    #    shutil.rmtree(params['event']['dir_data']+'Pickles_active/')
   
    if len(fireEvents)>0:
        gdf_activeEvent = create_gdf_fireEvents(fireEvents)
        gdf_to_gpkgfile(gdf_activeEvent, params, end_datetime, 'firEvents')
        gdf_to_gpkgfile(hsgdf_all_raw, params, end_datetime, 'hotspots')

    for id_, event in enumerate(fireEvents):
        if event is not None: 
            event.save( 'active', params, end_datetime)

    for id_, event in enumerate(pastFireEvents):
        event.save( 'past', params)

    print('number of events saved:')
    print('    active: {:d}'.format(count_not_none(fireEvents)))
    print('    past  : {:d}'.format(len(pastFireEvents)))

    return end_datetime, fireEvents, pastFireEvents

##############################################
def gdf_to_gpkgfile(gdf_activeEvent, params, datetime_, name_):
    tmp_path = "./{}-{}.gpkg".format(name_, datetime_.strftime("%Y-%m-%d_%H%M"))
    try:
        gdf_activeEvent.to_file(tmp_path, driver="GPKG")
    except: 
        pdb.set_trace()
    # Move to mounted share
    dst_path = os.path.join(params['event']['dir_data'], os.path.basename(tmp_path))
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
   
    #init dir
    #params = init('202504') 
    #start = datetime.strptime('2025-04-12_0000', '%Y-%m-%d_%H%M')
    #end = datetime.strptime('2025-04-15_0000', '%Y-%m-%d_%H%M')
    
    #params = init('AVEIRO') 
    #start = datetime.strptime('2024-09-15_0000', '%Y-%m-%d_%H%M')
    #end = datetime.strptime('2024-09-20_2300', '%Y-%m-%d_%H%M')
    
    params = init('SILEX') 
    start = datetime.strptime('2025-05-01_0000', '%Y-%m-%d_%H%M')
    end = datetime.strptime('2025-05-01_2300', '%Y-%m-%d_%H%M')
    #start = datetime.strptime('2025-05-01_2300', '%Y-%m-%d_%H%M')
    #end = datetime.strptime('2025-05-02_2300', '%Y-%m-%d_%H%M')

    # Loop hourly
    current = start
    while current <= end:

        #get last time processed
        if os.path.isfile("{:s}/hotspots-{:s}.gpkg".format(params['event']['dir_data'],(current+timedelta(hours=1)).strftime("%Y-%m-%d_%H%M")) ):
            print(current, end=' already done\n')
            current += timedelta(hours=1)
            continue 
        
        #track perimeter
        start_datetime = current.strftime('%Y-%m-%d_%H%M')
        end_datetime   = (current + timedelta(hours=1)).strftime('%Y-%m-%d_%H%M')
        date_now, fireEvents, pastFireEvents = perimeter_tracking(params, start_datetime,end_datetime)
        
        if date_now.hour == 20 and date_now.minute == 0:
            #ploting
            plot(params, date_now, fireEvents, pastFireEvents, flag_plot_hs=False, flag_remove_singleHs=True)

        #control hourly loop
        current += timedelta(hours=1)


    

