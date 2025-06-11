import sys
import os 
import pandas as pd
import geopandas as gpd
import shapely 
import numpy as np 
from matplotlib import pyplot as plt
from shapely.geometry import Polygon
from sklearn.cluster import AgglomerativeClustering
import multiprocessing
import pdb 
import warnings
#warnings.filterwarnings("error")
import pyproj
import importlib 
from rasterio.warp import calculate_default_transform, reproject, Resampling
import socket

#homebrwed
sys.path.append('../src-load/')
glc = importlib.import_module("load-glc-category")



########################
def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

########################
def get_dirData():
    if socket.gethostname() == 'europa':
        dir_data = '/home/paugam/Data/WII/'
    else:
        dir_data = '/mnt/dataEstrella/WII/'
    return dir_data


##########################
def my_read_file(filepath):
    if os.path.isfile(filepath.replace('.geojson','.prj')):
        tmp = gpd.read_file(filepath)
        with open(filepath.replace('.geojson','.prj'),'r') as f:
            lines = f.readlines()
        try:
            tmp.set_crs(crs=lines[0], allow_override=True, inplace=True)
        except: 
            pdb.set_trace()
        return tmp
    else:
        return gpd.read_file(filepath)


##########################
def cpu_count():
    try:
        return int(os.environ['ntask'])
    except:
        print('env variable ntask is not defined')
        sys.exit() 
        #return multiprocessing.cpu_count()

##########################
def cluster_shapes_by_distance(geodf, distance):
    """
    Make groups for all shapes within a defined distance. For a shape to be 
    excluded from a group, it must be greater than the defined distance
    from *all* shapes in the group.
    Distances are calculated using shape centroids.

    Parameters
    ----------
    geodf : data.frame
        A geopandas data.frame of polygons. Should be a projected CRS where the
        unit is in meters. 
    distance : float
        Maximum distance between elements. In meters.

    Returns
    -------
    np.array
        Array of numeric labels assigned to each row in geodf.

    """
    assert geodf.crs.is_projected, 'geodf should be a projected crs with meters as the unit'
    centers = [p.centroid for p in geodf.geometry]
    centers_xy = [[c.x, c.y] for c in centers]
    
    cluster = AgglomerativeClustering(n_clusters=None, 
                                      linkage='single',
                                      metric='euclidean',
                                      distance_threshold=distance)
    cluster.fit(centers_xy)
    
    return cluster.labels_


##########################
def dissolveGeometryWithinBuffer(gdf,bufferSize = 20.):

    gdf['geometry'] = gdf.geometry.apply(lambda g: g.buffer(bufferSize))

    s_ = gpd.GeoDataFrame(geometry=[gdf.unary_union]).explode( index_parts=False ).reset_index( drop=True )

    s_ = s_.geometry.apply(lambda g: g.buffer(-1*bufferSize))

    return s_

##########################
def getDistanceBetweenGdf(gdf1,gdf2):
    return gdf1.geometry.apply(lambda g: gdf2.distance(g))


##########################
def test_getDistanceBetweenGdf():
    df1 = pd.DataFrame(
        {
            "x": [0, 1, 1, 0],
            "y": [0, 0, 1, 1],
            "label": ['1', '1', '1', '1'],
        })

    df2 = pd.DataFrame(
        {
            "x": [3, 4, 4, 3],
            "y": [0, 0, 1, 1],
            "label": ['1', '1', '1', '1'],
        })
    gdf = gpd.GeoDataFrame(index=[1,2],
        geometry=[Polygon(zip(df1['x'],df1['y'])), Polygon(zip(df2['x'],df2['y']))]
        )
    print( getDistanceBetweenGdf(gdf,gdf))

    gdf.plot()
    plt.show()


##########################
def dist2FuelCat(indir,fuelCat, indus):
    
    bb = 1.e4 # 10 km
    nbregroup = indus['group'].max() +1
    for ig in range(0, nbregroup):
        #print('group {:d}/{:d} ... '.format(ig,nbregroup),end='\r')
        indus_ = indus.loc[indus['group']==ig]
        #print(indus_.shape)
        xmin, ymin, xmax, ymax = indus_.total_bounds
        
        fuelCat_ = fuelCat.cx[xmin-bb:xmax+bb, ymin-bb:ymax+bb]
        #print('sub fuelCat size', fuelCat_.shape)
        #fuelCat_ = dissolveGeometryWithinBuffer(fuelCat_,20) #only important for osm fuel data
        #print('compute distance')
        dist2fuelCat_ = getDistanceBetweenGdf(indus_,fuelCat_)
        pdb.set_trace()
        #fuelCat_ = None
        

        if ig == 0: 
            dist2fuelCat = dist2fuelCat_
        else:
            dist2fuelCat = pd.concat([dist2fuelCat, dist2fuelCat_])
    #print('done                 ') 
  
    mindist = dist2fuelCat.min(axis=1).sort_index()
    minPolyIdx = dist2fuelCat.idxmin(axis=1).sort_index()

    return mindist, minPolyIdx

##########################
def buildWII(WII, iv, fuelCat, indus, continent):

    importlib.reload(glc)
   
    bufferDistVegCat = [2400,1200,800,600,480,400,343]
    bb = 10.e3 
    nbregroup = indus['group'].max() +1
    #print('fuelCat{:d} - nbre group = {:d}'.format(iv,nbregroup))
    if  type(nbregroup) is not np.int64: pdb.set_trace()
    for ig in range(0, nbregroup):
        print('fuelCat{:d} - group {:d}/{:d} ... '.format(iv,ig,nbregroup),end='\r')
        sys.stdout.flush()
        indus_ = indus.loc[indus['group']==ig]
        #print(indus_.shape)
        xmin, ymin, xmax, ymax = indus_.total_bounds
        
        if type(fuelCat) is gpd.geodataframe.GeoDataFrame:
            fuelCat_ = fuelCat.cx[xmin-bb:xmax+bb, ymin-bb:ymax+bb]
        else: 
            indir = '{:s}CLC/'.format(get_dirData())
            to_latlon = pyproj.Transformer.from_crs(indus_.crs, 'epsg:4326')
            if np.abs(xmax-xmin) < 30.e3: bbx = 15.e3 
            else: bbx = bb
            if np.abs(ymax-ymin) < 30.e3: bby = 15.e3 
            else: bby = bb

            #lowerCorner = to_latlon.transform(ymin-bbx, xmin-bbx )
            #upperCorner = to_latlon.transform(ymax+bby, xmax+bby )
            
            lowerCorner = to_latlon.transform(xmin-bbx, ymin-bbx )
            upperCorner = to_latlon.transform(xmax+bby, ymax+bby )

            fuelCat_ = glc.clipped_fuelCat_gdf(indir, iv, indus_.crs, lowerCorner[1], lowerCorner[0], upperCorner[1], upperCorner[0])
            if fuelCat_ is None:
                continue
            else: 
                fuelCat_  =  add_AI2gdf(fuelCat_,ptdx=100,dbox=1000,PoverA=0.05)
   
        for iai in range(3):
            if iai == 0: 
                fuelCat__ = fuelCat_[(fuelCat_['AI']>0.9)                      ]
            elif iai == 1: 
                fuelCat__ = fuelCat_[(fuelCat_['AI']>0  )&(fuelCat_['AI']<=0.9)]
            elif iai == 2: 
                fuelCat__ = fuelCat_[(fuelCat_['AI']<=0 )                      ]  # to update to pass it to ==0. need PoverA update. 
        
            vegCat = iv + iai -1
            bufferDistVegCat_ = bufferDistVegCat[vegCat]

            indus__ = indus_.copy()
            indus__['geometry'] = indus_.geometry.apply(lambda g: g.buffer(bufferDistVegCat_))
       
            if len(fuelCat__)>0:
                WII_ = gpd.overlay(fuelCat__, indus__, how = 'intersection', keep_geom_type=False)
            else: 
                continue 

            if WII is None: 
                WII = WII_
            elif WII_.shape[0]>0:
                WII = pd.concat([WII, WII_])
   
        #ax = plt.subplot(111)
        #WII.plot(ax=ax)
        #indus_.plot(ax=ax,color='k')
        #plt.show()
        #pdb.set_trace()

    return WII


##########################
def star_AIpoly(param):
    return AIpoly(*param)

def AIpoly(gdf, ipo, ptdx = 100, dbox = 1000., PoverA=0.05):

    poly = gdf[ipo:ipo+1]
    
    minx,miny,maxx,maxy = poly.total_bounds
    xx = np.arange(minx,maxx+ptdx,ptdx)
    yy = np.arange(miny,maxy+ptdx,ptdx)
    ptsy, ptsx = np.meshgrid(yy,xx)
    
    gpts = gpd.GeoDataFrame(crs=poly.crs, geometry=[shapely.geometry.Point(x,y) for x,y in zip(ptsx.flatten(), ptsy.flatten())] )
    
    gpts2 = gpd.sjoin(gpts, poly, predicate = 'within')
    
    if gpts2.shape[0] == 0: 
        return 0 # poylgon was not catched by ptdx resolution, assume it is too small to have effect

    boxminx, boxminy, boxmaxx, boxmaxy = gpts2.total_bounds
    gdf_local = gdf.cx[boxminx-2*dbox:boxmaxx+2*dbox, boxminy-2*dbox:boxmaxy+2*dbox]

    AIpt = []
    for ipt in range(gpts2.shape[0]):
        
        pt = gpts2[ipt:ipt+1]
        
        x_= pt.geometry.iloc[0].coords.xy[0][0]
        y_= pt.geometry.iloc[0].coords.xy[1][0]
        boxx = [x_-dbox/2, x_+dbox/2, x_+dbox/2, x_-dbox/2,x_-dbox/2]
        boxy = [y_-dbox/2, y_-dbox/2, y_+dbox/2, y_+dbox/2,y_-dbox/2]
        box = gpd.GeoDataFrame(crs=poly.crs, geometry=[ shapely.geometry.Polygon(zip(boxx,boxy)) ] )

        gdf_box = gpd.overlay(gdf_local, box, how = 'intersection', keep_geom_type=False)
        
        Abox = (dbox**2)*1.e-4

        A    = gdf_box.area.sum()/1.e4   # total area ha of the selected polygon
        if A > .8*Abox : 
            AIpt.append(1)
            break 
        elif A < .1*Abox : 
            AIpt.append(0)
        else:
            P    = gdf_box.length.sum() * 1.e-3 # total perimeter km of the selected polygon
            #PoverA = 0.05 #max([0.04,P/A])
            A_unit = ptdx**2 * 1.e-4# ha
            S_unit = PoverA * A_unit*1.e4 #  m   square S_unit = 0.04 A_unit
            Pmax = S_unit * ( A /(A_unit) ) * 1.e-3
            Pmin = 2 * np.pi * np.sqrt(1.e4*A/np.pi) / 1.e3
            AIpt.append( 1 - (P-Pmin)/(Pmax-Pmin) ) #Boegart et al 2002 https://sites.bu.edu/cliveg/files/2013/12/jbogaert02.pdf

            #if P>Pmax : print(AIpt[-1], Pmax, P, Pmin, A/Abox, P/A*1.e-1)

            if AIpt[-1] == 1: break

    return max(AIpt)


##########################
def add_AI2gdf(gdf,ptdx,dbox,PoverA=0.05):
    
    params = []
    for ipo in range(gdf.shape[0]):
        params.append([gdf,ipo,ptdx,dbox,PoverA]) 

    flag_parallel_ = True
    if flag_parallel_:
        # set up a pool to run the parallel processing
        cpus = cpu_count()
        pool = multiprocessing.Pool(processes=cpus)

        # then the map method of pool actually does the parallelisation  
        results = pool.map(star_AIpoly, params)
        pool.close()
        pool.join()
       
    else:
        results = []
        for param in params:
            print('{:6.2f}%  '.format(100.*param[1]/gdf.shape[0]), end=' ')
            #if 100.*param[1]/gdf.shape[0] < 12: 
            #    print('',end='\r')
            #    continue
            results.append( star_AIpoly(param) )
            print('{:6.4f}'.format(results[-1]), end='\r')
            sys.stdout.flush()

 
    gdf['AI'] = results   

    return gdf


###########################################################
def reproject_raster(src_band, src_bounds, src_transform, src_crs, dst_crs, resolution=200):
    dst_transform, width, height = calculate_default_transform(
        src_crs,
        dst_crs,
        src_band.shape[1],
        src_band.shape[2],
        *src_bounds,  # unpacks outer boundaries (left, bottom, right, top)
        resolution=resolution
    )
    dst_band = np.zeros([height, width])

    try: 
        return  reproject(
        source=src_band,
        destination=dst_band,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=-999,
        resampling=Resampling.nearest)
    except:
        pdb.set_trace()


##########################
if __name__ == '__main__':
##########################

    gdf = gpd.read_file('./gdf-testCat1.geojson')
    dbox = 1000.
    ptdx = 100
    
    mm_ = add_AI2gdf(gdf,ptdx,dbox)
    plt.hist(mm_['AI'], 20, label='{:d}'.format(ptdx))
    
    plt.legend()
    plt.show()

    #indir = '/mnt/dataEstrella/OSM/FuelCategories/'
    #wood = gpd.read_file(indir+'wood.shp')

    #wood_geom = dissolveGeometryWithinBuffer(wood)

    #distancesIntraWood = getDistanceBetweenGdf(wood_geom,wood_geom)
    
