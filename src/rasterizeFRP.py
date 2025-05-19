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


#######################################################
def geojson2raster(filename, timestamp, lon, lat, flag_plot=False):

    ny, nx = lat.shape

    # Load fire events (GeoJSON)
    gdf = gpd.read_file(filename).to_crs("EPSG:4326")

    print(timestamp, gdf.time.max())    
    
    gdf = gdf.cx[lon.min():lon.max(),lat.min():lat.max()]
    if len(gdf)==0: 
        return None

    
    # Build cell polygons
    grid_polys = np.empty((ny - 1, nx - 1), dtype=object)
    for i in range(ny - 1):
        for j in range(nx - 1):
            corners = [
                (lon[i, j], lat[i, j]),
                (lon[i, j+1], lat[i, j+1]),
                (lon[i+1, j+1], lat[i+1, j+1]),
                (lon[i+1, j], lat[i+1, j])
            ]
            grid_polys[i, j] = Polygon(corners)

    # Prepare output array
    frp_grid = np.zeros((ny - 1, nx - 1), dtype=np.float32)

    # For each fire polygon, distribute FRP by overlap area
    for _, row in gdf.iterrows():
        fire_geom = row.geometry
        fire_frp = row["frp"]

        # Keep track of overlapping grid cells and areas
        overlap_weights = []
        total_overlap_area = 0
        for i in range(ny - 1):
            for j in range(nx - 1):
                cell_poly = grid_polys[i, j]
                if isinstance(fire_geom, Polygon) : 
                    if fire_geom.intersects(cell_poly):
                        inter_area = fire_geom.intersection(cell_poly).area
                        if inter_area > 0:
                            overlap_weights.append((i, j, inter_area))
                            total_overlap_area += inter_area
                if isinstance(fire_geom, LineString): 
                    fire_geom = fire_geom.buffer(0.0005)  # ~50 meters in degrees
                    if fire_geom.intersects(cell_poly):
                        inter_area = fire_geom.intersection(cell_poly).area
                        if inter_area > 0:
                            overlap_weights.append((i, j, inter_area))
                            total_overlap_area += inter_area
                if isinstance(fire_geom, Point): 
                    if cell_poly.contains(fire_geom):
                        inter_area = cell_poly.area
                        overlap_weights.append((i, j, inter_area))
                        total_overlap_area += inter_area

        # Distribute FRP proportionally to overlap area
        if total_overlap_area > 0:
            for i, j, area in overlap_weights:
                frp_grid[i, j] += fire_frp * (area / total_overlap_area)


    lat_center = 0.25 * (lat[:-1, :-1] + lat[1:, :-1] + lat[:-1, 1:] + lat[1:, 1:])
    lon_center = 0.25 * (lon[:-1, :-1] + lon[1:, :-1] + lon[:-1, 1:] + lon[1:, 1:])

    # Convert to seconds since epoch as int64
    dt64 = np.datetime64(gdf.time.max().tz_localize(None))
    seconds_since_epoch = (dt64 - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, 's')
    seconds_since_epoch = float(seconds_since_epoch)  # or .astype('int64')

    # Add a new time dimension (size 1)
    frp_da = xr.DataArray(
                frp_grid[np.newaxis, :, :],  # add new axis for time
                coords={
                    "time": [ seconds_since_epoch ],
                    "lat": (["y", "x"], lat_center),
                    "lon": (["y", "x"], lon_center)
                },
                dims=["time", "y", "x"],
                name="frp"
            )
    try:
        assert np.isclose(frp_da.sum().item(), gdf["frp"].sum(), rtol=1e-2)
    except: 
        pdb.set_trace()

    if flag_plot:
        #plot
        lon2d = frp_da.lon.values  # shape (ny, nx)
        lat2d = frp_da.lat.values
        frp = frp_da.values

        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})

        # Plot FRP raster (curvilinear grid)
        mesh = ax.pcolormesh(
            lon2d, lat2d, frp,
            shading='auto',  # ensures correct handling of curvilinear grids
            cmap='hot', vmin=0, vmax=np.nanmax(frp), alpha=0.8
        )

        # Add fire polygons from GeoDataFrame
        gdf.plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=1)

        # Add coastlines and borders for context
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')

        # Colorbar and labels
        cbar = plt.colorbar(mesh, ax=ax, label="FRP (MW)")
        ax.set_title("FRP Raster on Curvilinear Grid with Fire Polygons")
        plt.tight_layout()
        plt.show()


    return frp_da


#########################
if __name__== '__main__':
#########################

# def path
    dirData = '/mnt/dataEstrella2/SILEX/'
    dirFireEvent = dirData+'VIIRS-HotSpot/FireEvents/GeoJson/'
    geojsons =sorted(glob.glob(dirFireEvent+'*.geojson'))

# Load the NetCDF grid, bottom left corner location
    input_domain = dirData+"MNH/PGD_D2000mA.nested.nc"
    grid = xr.open_dataset(input_domain)
    domain_name = os.path.basename(input_domain).split('.')[0]
    lon = grid["longitude_u"].values
    lat = grid["latitude_v"].values

#define start end time
    start_time = np.datetime64("2025-05-18T04:00:00") #assume UTC
    end_time   = np.datetime64("2025-05-18T19:00:00")

    frp_list = []
    for geojson in geojsons:
        
        match = re.search(r'firEvents-(\d{4}-\d{2}-\d{2})_(\d{4})', os.path.basename(geojson))
        date_part, time_part = match.groups()
        timestamp = np.datetime64(f"{date_part}T{time_part[:2]}:{time_part[2:]}:00")
       
        if not (start_time <= timestamp <= end_time): continue
        
        frp_da = geojson2raster(geojson,timestamp, lon, lat )
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
    frp_series.to_netcdf(f"{dirData}/MNH/frp_{domain_name}_{str_starttime}-{str_endtime}.nc")

