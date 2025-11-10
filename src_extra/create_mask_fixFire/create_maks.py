import os
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation
import xarray as xr
import rioxarray  # important: registers the ".rio" accessor

if __name__ == '__main__':

    root_data                         = '/data/shared/'
    file_HSDensity_ESAWorldCover      = root_data + 'HotspotDensity2024/HSDensity_med/hs_density_med.vrt'
    threshold_HSDensity_ESAWorldCover = 11 
    file_polygonIndus_OSM             = '/data/paugam/OSM_IndustrialZone/industrialZone_osmSource-med.geojson'
    root_data_local                   = '../../data_local/'
    out_mask_name = 'mask_hs_600m_med'

    maskHS_da = None
    print('generate mask for fix hs ...')
    print('using VIIRS HS density from 2024 and polygon mask from OSM')
    with rasterio.open(file_HSDensity_ESAWorldCover) as src:
        HSDensity = src.read(1, masked=True)  # Use masked=True to handle nodata efficiently
        transform = src.transform
        crs = src.crs
        threshold = threshold_HSDensity_ESAWorldCover

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
        )
    maskHS_da.rio.write_crs(crs, inplace=True)
    
    #add OSM industrial polygon to the mask
    indusAll = gpd.read_file(file_polygonIndus_OSM).to_crs(crs) 
   
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

    maskHS_da.to_netcdf(f'{root_data_local}{out_mask_name}.nc')
