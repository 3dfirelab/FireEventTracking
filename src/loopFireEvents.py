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
from shapely.geometry import Point, MultiPoint, mapping, Polygon, MultiPolygon, LineString 
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
from shapely.geometry.polygon import orient
import json
from scipy.interpolate import griddata
import re 

#home brewed
import fireEvent as fireEventTool
import fireEventTracking
import interpArrivalTime
import tools4FOREFIRE as tools4forefire


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


#######################################
def open_nc_before_date(dir_FWI, suffix, reference_datetime):
    """
    Open the *_suffix.nc file with the most recent timestamp before a given reference datetime.

    Parameters:
        dir_FWI (str or Path): Directory containing NetCDF files.
        suffix (str): File suffix to match (e.g., "fwiffmc").
        reference_datetime (datetime or pandas.Timestamp): UTC datetime to search before.

    Returns:
        xarray.Dataset: Dataset of the closest matching file before the reference datetime.
    """
    dir_FWI = Path(dir_FWI)
    pattern = re.compile(r"(\d{{8}}\.\d{{2}})Z_{}\.(nc)$".format(re.escape(suffix)))
    files = []

    # Convert to naive UTC datetime if Timestamp is timezone-aware
    if isinstance(reference_datetime, pd.Timestamp) and reference_datetime.tzinfo is not None:
        reference_datetime = reference_datetime.tz_convert(None)

    for fname in dir_FWI.iterdir():
        match = pattern.search(fname.name)
        if match:
            try:
                dt = datetime.strptime(match.group(1), "%Y%m%d.%H")
                if dt < reference_datetime:
                    files.append((dt, fname))
            except ValueError:
                continue

    if not files:
        raise FileNotFoundError(f"No *_{suffix}.nc files found before {reference_datetime} in {dir_FWI}")

    most_recent_file = max(files, key=lambda x: x[0])[1]
    return xr.open_dataset(most_recent_file)


###########################
def clip_dataset_with_buffer(dsin, bbox_wsen, buffer_deg=1.0):
    """
    Clip an xarray dataset with a buffered bounding box.

    Parameters:
        dsin (xarray.Dataset): Input dataset with 'lat' and 'lon' variables (2D).
        bbox_wsen (tuple): Bounding box (W, S, E, N).
        buffer_deg (float): Buffer in degrees to expand the bounding box.

    Returns:
        xarray.Dataset: Clipped dataset.
    """
    west, south, east, north = bbox_wsen
    west_b, south_b, east_b, north_b = west - buffer_deg, south - buffer_deg, east + buffer_deg, north + buffer_deg

    # Ensure the dataset has 2D lat/lon
    if not (("lat" in dsin.variables) and ("lon" in dsin.variables)):
        raise ValueError("Dataset must contain 'lat' and 'lon' variables.")

    lats = dsin['lat']
    lons = dsin['lon']

    mask = ((lons >= west_b) & (lons <= east_b) &
            (lats >= south_b) & (lats <= north_b))

    return dsin.where(mask, drop=True)


############################
def project_to_ffgrid(dsin, varname, fflon, fflat,  ):
    # Get source points and values
    lon_src = dsin['lon'].values
    lat_src = dsin['lat'].values

    # Create 2D meshgrid from 1D lat/lon
    lon2d, lat2d = np.meshgrid(lon_src, lat_src)

    # Prepare source points and values for interpolation
    points = np.column_stack((lon2d.ravel(), lat2d.ravel()))
   
    # Interpolate onto target grid
    var_on_ff = []
    fwi_vals = dsin[varname].values
    for it in range(dsin.sizes['time']):
        values = fwi_vals[it].ravel()
        var_on_ff.append( griddata(points, values, (fflon, fflat), method='nearest') )

    return np.array(var_on_ff)
           

#############################
def ffmc_to_fmc(ffmc):
    """
    Convert Canadian FFMC code to FMC (% by dry weight) using Van Wagner (1987).
    
    Parameters:
        ffmc (float or array-like): Fine Fuel Moisture Code (0–101)
    
    Returns:
        float or array-like: FMC in percent
    """
    return 147.2 * (101 - ffmc) / (59.5 + ffmc)   



#############################
def make_geojson_from_geom(geom, time_igni):
    """
    Create a GeoJSON FeatureCollection from a shapely geometry, supporting Point, LineString, Polygon, and MultiPolygon.
    
    Parameters:
        geom (shapely geometry): One of Point, LineString, Polygon, or MultiPolygon.
        time_igni (datetime): Timestamp for 'valid_at' field.
        
    Returns:
        dict: GeoJSON FeatureCollection.
    """
    if isinstance(geom, Point):
        coordinates = [geom.x, geom.y, 0]
        geometry = {
            "type": "Point",
            "coordinates": coordinates
        }

    elif isinstance(geom, LineString):
        coordinates = [[x, y, 0] for x, y in geom.coords]
        geometry = {
            "type": "LineString",
            "coordinates": coordinates
        }

    elif isinstance(geom, Polygon):
        def polygon_to_3d_coords(polygon):
            exterior = [[x, y, 0] for x, y in polygon.exterior.coords]
            interiors = [[[x, y, 0] for x, y in ring.coords] for ring in polygon.interiors]
            return [exterior] + interiors
        
        multipolygon_coords = [polygon_to_3d_coords(geom)]
        geometry = {
            "type": "MultiPolygon",
            "coordinates": multipolygon_coords
        }

    elif isinstance(geom, MultiPolygon):
        def polygon_to_3d_coords(polygon):
            exterior = [[x, y, 0] for x, y in polygon.exterior.coords]
            interiors = [[[x, y, 0] for x, y in ring.coords] for ring in polygon.interiors]
            return [exterior] + interiors
        
        multipolygon_coords = [polygon_to_3d_coords(p) for p in geom.geoms]
        geometry = {
            "type": "MultiPolygon",
            "coordinates": multipolygon_coords
        }

    else:
        raise ValueError(f"Unsupported geometry type: {type(geom)}")

    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": {
                    "valid_at": time_igni.strftime("%Y-%m-%dT%H:%M:%SZ")
                }
            }
        ]
    }


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
        if fireEvent.id_fire_event != 7788 : continue 
   
        out_dir_event = '{:s}/Pickles_{:s}_{:s}/FOREFIRE/{:09d}/'.format(params['event']['dir_data'],'active',
                                                         currentTime.strftime("%Y-%m-%d_%H%M"),fireEvent.id_fire_event)
        os.makedirs(out_dir_event, exist_ok=True)
        firename = '{:d}-{:s}'.format(fireEvent.id_fire_event,fireEvent.times[-1].strftime('%Y%m%d.%H%M'))
    
        igni = {}
        igni['time'] = fireEvent.times[-1]
        igni['perimeter'] = fireEvent.ctrs.to_crs(4326).iloc[-1]
        

        #1. collect input data to create tile
        #------------------------------------
        #get initial file from corte.
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
        
        #2. add wind and fmc from arome and fwi calculation
        #--------------------
        # Projected grid coordinates
        dx = 1000. #ds['domain'].attrs['Lx'] / ds.dims['fx']
        dy = 1000. #ds['domain'].attrs['Ly'] / ds.dims['fy']
        SWx = ds['domain'].attrs['SWx']
        SWy = ds['domain'].attrs['SWy']
        nx = int(ds['domain'].attrs['Lx']/dx)
        ny = int(ds['domain'].attrs['Ly']/dy)

        x = SWx + dx * np.arange(nx)  
        y = SWy + dy * np.arange(ny)  

        # Create 2D meshgrid of projected coordinates
        X, Y = np.meshgrid(x, y)  

        # Initialize projection with WSENLBRT
        proj = tools4forefire.projForeFire( np.array( ds['domain'].attrs['WSENLBRT'].split(','), dtype=float) ) 

        # Get lon/lat grids
        fflon, fflat = proj.xy_to_lonlat(X, Y)  # Both (fy, fx)
        
        #for ffmc
        dsin = open_nc_before_date( params['event']['dir_FWI'], 'fwiffmc', igni['time'])
        time_ffmc = dsin.time.values
        dsin_clipped = clip_dataset_with_buffer(dsin, np.array(ds['domain'].attrs['BBoxWSEN'].split(','),dtype=float), buffer_deg=0.25)
        ffmc_ff = project_to_ffgrid(dsin_clipped, 'FFMC', fflon, fflat  )
        moisture = ffmc_to_fmc(ffmc_ff)
        
        #for wind
        dsin = open_nc_before_date( params['event']['dir_WIND'], 'wind', igni['time'])
        time_wind = dsin.time.values
        dsin_clipped = clip_dataset_with_buffer(dsin, np.array(ds['domain'].attrs['BBoxWSEN'].split(','),dtype=float), buffer_deg=0.25)
        wind_u_ff = project_to_ffgrid(dsin_clipped, 'wind_u', fflon, fflat  )
        wind_v_ff = project_to_ffgrid(dsin_clipped, 'wind_v', fflon, fflat  )
    
        #add two new data in ds
        # Assume moisture is (T, Y, X)
        # Add singleton Z-dimension
        moisture_da = xr.DataArray(
            moisture[:, np.newaxis, :, :].astype(np.float32),  # (T, Z=1, Y, X)
            dims=("atm_t", "atm_z", "atm_y", "atm_x"),
            name="moisture",
            attrs={"type": "data"}
        )

        windU_da = xr.DataArray(
            wind_u_ff[:, np.newaxis, :, :].astype(np.float32),
            dims=("atm_t", "atm_z", "atm_y", "atm_x"),
            name="windU",
            attrs={"type": "data"}
        )

        windV_da = xr.DataArray(
            wind_v_ff[:, np.newaxis, :, :].astype(np.float32),
            dims=("atm_t", "atm_z", "atm_y", "atm_x"),
            name="windV",
            attrs={"type": "data"}
        )

        # Merge variables into the dataset
        ds = ds.assign(
            moisture=moisture_da,
            windU=windU_da,
            windV=windV_da
        )
        #clean up unused var and dim
        ds = ds.drop_vars("wind")

        #adjust time
        ds.domain.attrs['Lt'] =np.float32((time_wind[-1] - time_wind[0]) / np.timedelta64(1, 's'))

        #save
        dst_filename = out_dir_event+'data.nc'
        ds.to_netcdf(dst_filename)
        

        #3. create ignition geojson
        #-----------------------
        poly = fireEvent.ctrs.to_crs(4326).iloc[-1].geometry
        if isinstance(poly, Polygon):
            poly = orient(poly, sign=1.0)
        
        geojson_dict = make_geojson_from_geom(poly, igni['time'])
        # Write to file
        with open(f"{out_dir_event}front.geojson", "w") as f:
            json.dump(geojson_dict, f, indent=2)

        pdb.set_trace() 
        
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
    importlib.reload(tools4forefire)
    dirDataTileFOREFIRE = '/mnt/dataEstrella2/FOREFIRE_TILE'

    #init
    params = fireEventTracking.init('SILEX','../log/') 
    currentTime = datetime.strptime('2025-06-07_1500', '%Y-%m-%d_%H%M')
   
    #loop for FOREFIRE
    loopFireEvents4FireGrowth(params, currentTime )

    #loop for ROS
    #loopFireEvents4ROS(params, currentTime )



