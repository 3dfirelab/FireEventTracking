import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString, Polygon
import glob
import sys
import xarray as xr 
import re 
import matplotlib.pyplot as plt
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os 
import pdb 
import pandas as pd 
import importlib 
from shapely.strtree import STRtree


#homebrewed 
import rasterizeFRP_MNH as rasterizeFET


#########################
if __name__== '__main__':
#########################
    
    importlib.reload(rasterizeFET)

# def path
    dirData = '/mnt/dataEstrella2/SILEX/'
    dirFireEvent = dirData+'VIIRS-HotSpot/FireEvents/GeoJson/'
    geojsons =sorted(glob.glob(dirFireEvent+'*.geojson'))

    os.makedirs(dirData+'MOCAGE/',exist_ok=True)

# Load the NetCDF grid, bottom left corner location
    domain_name = 'MOCAGE'
    lon = np.arange(-26, 46+0.1, 0.1)
    lat = np.arange(29,73+0.1,0.1)
    lats, lons = np.meshgrid(lat,lon)

#define start end time
    start_time = np.datetime64("2025-05-18T04:00:00") #assume UTC
    end_time   = np.datetime64("2025-06-02T23:50:00")
    
    # Build cell polygons
    nx, ny = lats.shape
    grid_polys = np.empty((nx - 1, ny - 1), dtype=object)
    for i in range(nx - 1):
        for j in range(ny - 1):
            corners = [
                (lons[i, j],     lats[i, j]),
                (lons[i, j+1],   lats[i, j+1]),
                (lons[i+1, j+1], lats[i+1, j+1]),
                (lons[i+1, j],   lats[i+1, j])
            ]
            grid_polys[i, j] = Polygon(corners)


    # Flatten grid_polys and keep track of indices
    #flat_grid_polys = grid_polys.flatten()
    #indices = np.arange(flat_grid_polys.shape[0])
    #tree = STRtree(flat_grid_polys)
    #index_map = dict((id(geom), idx) for idx, geom in enumerate(flat_grid_polys))
    #flat_grid = grid_polys.reshape(-1)
    #ij_array = np.array([np.unravel_index(k, grid_polys.shape) for k in range(flat_grid.size)])
    #tree = STRtree(flat_grid)
    

# Flatten grid_polys and index tracking
    flat_grid_polys = grid_polys.flatten()
    ij_list = np.array([np.unravel_index(i, grid_polys.shape) for i in range(flat_grid_polys.size)])

# Build STRtree
    tree = STRtree(flat_grid_polys)

# Build reverse map: geometry -> (i, j)
# Works for Shapely 2.x using .geometries
    geom_to_ij = {geom: ij for geom, ij in zip(tree.geometries, ij_list)}


    frp_list = []
    for geojson in geojsons:
        
        match = re.search(r'firEvents-(\d{4}-\d{2}-\d{2})_(\d{4})', os.path.basename(geojson))
        date_part, time_part = match.groups()
        timestamp = np.datetime64(f"{date_part}T{time_part[:2]}:{time_part[2:]}:00")
       
        if not (start_time <= timestamp <= end_time): continue
        
        frp_da = rasterizeFET.geojson2raster(geojson,timestamp, lons, lats, grid_polys, tree, geom_to_ij)
        if frp_da is None: continue
        frp_list.append(frp_da)

# Concatenate all time slices and set attrs
    frp_series = xr.concat(frp_list, dim="time")
    frp_series.attrs["long_name"] = "Fire Radiative Power"
    frp_series.attrs["units"] = "MW"
    frp_series.attrs["description"] = "Rasterized VIIRS FRP on curvilinear model grid"

    frp_series.coords["lat"].attrs["standard_name"] = "latitude"
    frp_series.coords["lat"].attrs["units"] = "degrees_north"
    frp_series.coords["lat"].attrs["description"] = "cell center latitude"

    frp_series.coords["lon"].attrs["standard_name"] = "longitude"
    frp_series.coords["lon"].attrs["units"] = "degrees_east"
    frp_series.coords["lon"].attrs["description"] = "cell center  longitude"
    
    frp_series["time"].attrs["units"] = "seconds since 1970-01-01 00:00:00"
    frp_series["time"].attrs["dtype"] = "float64"
    frp_series["time"].attrs["calendar"] = "standard"

#save to netcdf
    str_starttime = pd.to_datetime(start_time).strftime("%Y%m%dT%H%M")
    str_endtime = pd.to_datetime(end_time).strftime("%Y%m%dT%H%M")
    frp_series.to_netcdf(f"{dirData}/MOCAGE/frp_{domain_name}_{str_starttime}-{str_endtime}.nc")

