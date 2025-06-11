import sys
import os 
#os.environ['USE_PYGEOS'] = '0'
import pandas as pd 
import geopandas as gpd
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
import pyproj
import rasterio
import shapely
from fiona.crs import from_epsg
from rasterio.mask import mask
import pdb 
from multiprocessing import Pool, cpu_count
import warnings
import json

#homebrewed
sys.path.append('../src-map/')
import tools


def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    return [ json.loads(gdf.to_json())['features'][ii]['geometry']  for ii in range(len(gdf)) ]

def clipped_fuelCat_raster(indir, iv, crs_here, xminContinent,yminContinent, xmaxContinent,ymaxContinent, bordersSelection):
    warnings.simplefilter(action='ignore', category=FutureWarning)
        
    to_latlon = pyproj.Transformer.from_crs(crs_here, 'epsg:4326')

    lowerCorner = to_latlon.transform(yminContinent, xminContinent)
    upperCorner = to_latlon.transform(ymaxContinent, xmaxContinent)

    src_bounds = (lowerCorner[1], lowerCorner[0], upperCorner[1], upperCorner[0])
    #print('fuel global lc')
    fuelCatTag = []
    fuelCatTag.append([111,113,121,123]) #1
    fuelCatTag.append([115,116,125,126]) #2
    fuelCatTag.append([112,114,122,124,20,30]) #3
    fuelCatTag.append([90]) #4
    fuelCatTag.append([100]) #5

    filein = indir + 'PROBAV_LC100_global_v3.0.1_2018-conso_Discrete-Classification-map_EPSG-4326.tif' 

    gdflc = None
    with rasterio.open(filein) as src:
        
        #clip
        #bbox = shapely.geometry.box(lowerCorner[1], lowerCorner[0], upperCorner[1], upperCorner[0])
        bbox = gpd.clip(bordersSelection,(xminContinent,yminContinent, xmaxContinent,ymaxContinent)).explode().buffer(-.01).to_crs('epsg:4326').reset_index().drop(['level_0','level_1'], axis=1)
        
        geo = gpd.GeoDataFrame({'geometry': bbox[0].geometry}, index=range(len(bbox)),) 
        
        #geo = geo.to_crs(crs=src.crs.data)
        coords = getFeatures(geo)
        try:
            data_, src_transform = mask(src, shapes=coords, crop=True)
        except: 
            pdb.set_trace()
        data_out, transform_out = tools.reproject_raster(data_, src_bounds, src_transform, geo.crs, crs_here,)
        data_ = None
        
        rev = ~transform_out # inverse transformation
        jjmin, iimin = rev*(xminContinent,ymaxContinent)
        jjmax, iimax = rev*(xmaxContinent,yminContinent)
        data_out = data_out[int(np.round(iimin,0)):int(np.round(iimax,0)),int(np.round(jjmin,0)):int(np.round(jjmax,0))]
        
        #plt.imshow(data_out)
        #plt.show()
        #pdb.set_trace()

        #print ('fuelCat ', iv, end='')
        condition =  (data_out!=fuelCatTag[iv-1][0])
        if len(fuelCatTag[iv-1]) > 1:
            for xx in fuelCatTag[iv-1][1:]:
                condition &= (data_out!=xx)
        data_out_masked = np.ma.masked_where(condition,data_out)

    return data_out_masked


def clipped_fuelCat_gdf(indir, iv, crs, xminContinent,yminContinent, xmaxContinent,ymaxContinent):

    #print('fuel global lc')
    fuelCatTag = []
    fuelCatTag.append([111,113,121,123]) #1
    fuelCatTag.append([115,116,125,126]) #2
    fuelCatTag.append([112,114,122,124,20,30]) #3
    fuelCatTag.append([90]) #4
    fuelCatTag.append([100]) #5

    filein = indir + 'PROBAV_LC100_global_v3.0.1_2018-conso_Discrete-Classification-map_EPSG-4326.tif' 

    gdflc = None
    with rasterio.open(filein) as src:
     
        if yminContinent > src.bounds.top: 
            return gpd.GeoDataFrame()
        
        if ymaxContinent < src.bounds.bottom: 
            return gpd.GeoDataFrame()
        
        #clip
        bbox = shapely.geometry.box(xminContinent,
                                    max(yminContinent,src.bounds.bottom),
                                    xmaxContinent,
                                    min(ymaxContinent,src.bounds.top))
        geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=from_epsg(4326))
        #geo = geo.to_crs(crs=src.crs.data)
        coords = getFeatures(geo)
        try:
            data_, out_transform = mask(src, shapes=coords, crop=True)
        except: 
            pdb.set_trace()
        
        #print ('fuelCat ', iv, end='')
        condition =  (data_!=fuelCatTag[iv-1][0])
        if len(fuelCatTag[iv-1]) > 1:
            for xx in fuelCatTag[iv-1][1:]:
                condition &= (data_!=xx)
        data_masked = np.ma.masked_where(condition,data_)
       
        try:
            if not(False in data_masked.mask): return None
        except:
            data_masked.mask = np.zeros_like(data_)
            if not(False in data_masked.mask): return None
             
        #print (' -- array loaded')
        # Use a generator instead of a list
        shape_gen = ((shapely.geometry.shape(s), v) for s, v in rasterio.features.shapes(data_masked, transform=out_transform))

        # either build a pd.DataFrame
        # df = DataFrame(shape_gen, columns=['geometry', 'class'])
        # gdf = GeoDataFrame(df["class"], geometry=df.geometry, crs=src.crs)

        # or build a dict from unpacked shapes
        gdf = gpd.GeoDataFrame(dict(zip(["geometry", "class"], zip(*shape_gen))), crs=src.crs)

        return gdf.to_crs(crs)


def fctunion(x):
    return x.buffer(1).unary_union.buffer(-1)

if __name__ == '__main__':

    warnings.filterwarnings("ignore")
    continent = 'asia'
    
    if continent == 'europe':
        xminContinent,yminContinent, xmaxContinent,ymaxContinent = [-31., 24.505457173625324, 99.52727619009086, 80.51193780175987]
        crs_here = 'epsg:3035'
    elif continent == 'asia':
        #xminContinent,yminContinent, xmaxContinent,ymaxContinent = [28.7, -14.9, 150, 87.]
        xminContinent,yminContinent, xmaxContinent,ymaxContinent = [40, -15., 140, 50.]
        #xminContinent,yminContinent, xmaxContinent,ymaxContinent = [124.613617-1,  33.197577+1,  131.862522-1,  38.624335+1 ] 
        crs_here = 'epsg:3832'

    categories = np.arange(1,6)
    indir = '/mnt/dataEstrella/WII/CLC/'
    outdir = '/mnt/dataEstrella/WII/FuelCategories-CLC/{:s}/'.format(continent)

    dd = 0.5
    lonsTyle = np.arange(xminContinent,xmaxContinent+dd,dd)
    latsTyle = np.arange(yminContinent,ymaxContinent+dd,dd)
        
    nn = (lonsTyle.shape[0]-1)*(latsTyle.shape[0]-1)
    for iv in categories:
        print('fuelcat{:d}'.format(iv))
        ii = 0
        fuelCats = []
        for ilat, lat_ in enumerate(latsTyle[:-1]):
            for ilon, lon_ in enumerate(lonsTyle[:-1]):

                gdf = clipped_fuelCat_gdf(indir, iv, crs_here, lon_, lat_, lonsTyle[ilon+1], latsTyle[ilat+1])
                print('{:.2f}'.format(100.*ii/nn), end='\r')
               
                if gdf is None: 
                    ii += 1
                    continue
    
                if gdf.shape[0]>20:
                    geom_arr = []
                    for i in range(0, len(gdf), 10):
                        geom_arr.append(gdf.iloc[i:i+10])
             
                    with Pool(20) as p:
                        geom_union = p.map(fctunion, geom_arr) 
                
                    gdf = gpd.GeoDataFrame(geometry=geom_union, crs=crs_here).explode().reset_index()
                
                else:
                    geom_union = gdf.buffer(1).unary_union.buffer(-1)

                    gdf = gpd.GeoDataFrame(geometry=[geom_union], crs=crs_here).explode().reset_index()
                
                fuelCats.append(gdf)

                ii += 1

        gdf = pd.concat(fuelCats)
        #union = gdf.buffer(1).unary_union.buffer(-1)
        print('shape gdf :', gfd.shape)
        
        geom_arr = []
        # Converting geometries list to nested list of geometries
        for i in range(0, len(gdf), 10000):
            geom_arr.append(gdf.iloc[i:i+10000])
 
        # Creating multiprocessing pool to perform union operation of chunks of geometries
        with Pool(cpu_count()) as p:
            geom_union = p.map(fctunion, geom_arr) 

        gdf = gpd.GeoDataFrame(geometry=geom_union, crs=crs_here).explode().reset_index()

        gdf.to_file(outdir+'fuelCategory{:d}.geojson'.format(iv),driver='GeoJSON')


