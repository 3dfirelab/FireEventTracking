import sys
import os 
import pandas as pd
import geopandas as gpd
import shapely 
import glob
import matplotlib as mpl
from matplotlib import pyplot as plt
from shapely.geometry import Polygon
import importlib
import warnings
import pyproj
from fiona.crs import from_epsg
import pdb 
import argparse
import numpy as np 

#homebrewed
import tools
import params

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='map Industrial area')
    parser.add_argument('-c','--continent', help='continent name',required=True)
    args = parser.parse_args()  

    continent = args.continent
    #continent = 'asia'
    
    dir_data = '/data/shared/'
    dir_out_root = '/data/shared/OSM_IndustrialZone/'

    importlib.reload(tools)
    
    params = params.load_param(continent)
    xminAll,xmaxAll = params['xminAll'], params['xmaxAll']
    yminAll,ymaxAll = params['yminAll'], params['ymaxAll']
    crs_here        = params['crs_here']
    bufferBorder    = params['bufferBorder']
    lonlat_bounds   = params['lonlat_bounds']
    gratreso        = params['gratreso']

    #borders
    indir = '{:s}Boundaries/'.format(dir_data)
    if continent == 'europe':
        bordersNUTS = gpd.read_file(indir+'NUTS/NUTS_RG_01M_2021_4326.geojson')
        bordersNUST = bordersNUTS.to_crs(crs_here)
        extraNUTS = gpd.read_file(indir+'noNUTS.geojson')
        extraNUST = extraNUTS.to_crs(crs_here)
        bordersSelection = pd.concat([bordersNUST,extraNUST])
    else:
        bordersSelection = tools.my_read_file(indir+'mask_{:s}.geojson'.format(continent))
        bordersSelection = bordersSelection[['SOV_A3', 'geometry', 'LEVL_CODE']]
        bordersSelection = bordersSelection.dissolve(by='SOV_A3', aggfunc='sum').reset_index()
    bordersSelection = bordersSelection.to_crs(crs_here)

    landNE = gpd.read_file(indir+'NaturalEarth_10m_physical/ne_10m_land.shp')
    #load graticule
    #gratreso = 15
    graticule = gpd.read_file(indir+'NaturalEarth_graticules/ne_110m_graticules_{:d}.shp'.format(gratreso))

    if lonlat_bounds is not None:
        landNE_ = pd.concat( [ gpd.clip(landNE,lonlat_bounds_) for lonlat_bounds_ in lonlat_bounds])
        graticule_ = pd.concat( [ gpd.clip(graticule,lonlat_bounds_) for lonlat_bounds_ in lonlat_bounds])
    else: 
        landNE_ = landNE
        graticule_= graticule

    landNE = landNE_.to_crs(crs_here)
    graticule = graticule_.to_crs(crs_here)

    #industrial zon
    indir = '{:s}/{:s}/'.format(dir_out_root,continent)
    indusFiles = sorted(glob.glob(indir+'*.geojson'))
    
    dirout = '{:s}/'.format(dir_out_root)
    tools.ensure_dir(dirout)

    if not(os.path.isfile(f'{dirout}/industrialZone_osmSource-{continent}.geojson')):
        indusAll = None
        for indusFile in indusFiles:

            print(os.path.basename(indusFile))
            indus = gpd.read_file(indusFile)
            indus = indus.to_crs(crs_here)

            indus['area_ha'] = indus['geometry'].area/ 10**4
            indus = indus[indus['area_ha']>1]

            indus = gpd.overlay(indus, bordersSelection, how = 'intersection', keep_geom_type=False)

            if indusAll is None:
                indusAll = indus
            else: 
                indusAll = pd.concat([indusAll,indus])

        indusAll.to_file(f'{dirout}/industrialZone_osmSource-{continent}.geojson',driver='GeoJSON')
        if indusAll.crs.to_epsg() is None: 
            with open(f'{dirout}/industrialZone_osmSource-{continent}.prj','w') as f:
                f.write(indusAll.crs.to_wkt())
        '''
        need to find a trick to save crs for namerica as it is not an epsg code
        saving in geojson reset it to lalon WGS84
        '''
    else: 
        indusAll = tools.my_read_file(f'{dirout}/industrialZone_osmSource-{continent}.geojson')

    mpl.rcdefaults()
    mpl.rcParams['font.size'] = 14
    mpl.rcParams['xtick.labelsize'] = 14
    mpl.rcParams['ytick.labelsize'] = 14
    mpl.rcParams['figure.subplot.left'] = .1
    mpl.rcParams['figure.subplot.right'] = .95
    mpl.rcParams['figure.subplot.top'] = .9
    mpl.rcParams['figure.subplot.bottom'] = .05
   
    ratio_ = (ymaxAll-yminAll)/(xmaxAll-xminAll)
    fig = plt.figure(figsize=(10,(np.round(ratio_,1))*10+1))
    ax = plt.subplot(111)
    landNE.plot(ax=ax,facecolor='0.9',edgecolor='None',zorder=1)
    graticule.plot(ax=ax, color='lightgrey',linestyle=':',alpha=0.95,zorder=3)
    #bordersSelection.buffer(bufferBorder)[bordersSelection['LEVL_CODE']==0].plot(ax=ax,facecolor='0.75',edgecolor='None',zorder=2)

    indusAll.plot(ax=ax, facecolor='k', edgecolor='k', linewidth=.2,zorder=4)
    ax.set_xlim(xminAll,xmaxAll)
    ax.set_ylim(yminAll,ymaxAll)
    
    #set axis
    bbox = shapely.geometry.box(xminAll, yminAll, xmaxAll, ymaxAll)
    geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=indusAll.crs)
    geo['geometry'] = geo.boundary
    ptsEdge =  gpd.overlay(graticule, geo, how = 'intersection', keep_geom_type=False)
    
    lline = shapely.geometry.LineString([[xminAll,ymaxAll],[xmaxAll,ymaxAll]])
    geo = gpd.GeoDataFrame({'geometry': lline}, index=[0], crs=indusAll.crs)
    ptsEdgelon =  gpd.overlay(ptsEdge, geo, how = 'intersection', keep_geom_type=False)
    ptsEdgelon = ptsEdgelon[(ptsEdgelon['direction']!='N')&(ptsEdgelon['direction']!='S')]
    
    ax.xaxis.set_ticks(ptsEdgelon.geometry.centroid.x)
    ax.xaxis.set_ticklabels(ptsEdgelon.display, rotation=33)
    ax.xaxis.tick_top()
    
    lline = shapely.geometry.LineString([[xminAll,yminAll],[xminAll,ymaxAll]])
    geo = gpd.GeoDataFrame({'geometry': lline}, index=[0], crs=indusAll.crs)
    ptsEdgelat =  gpd.overlay(ptsEdge, geo, how = 'intersection', keep_geom_type=False)
    ptsEdgelat = ptsEdgelat[(ptsEdgelat['direction']!='E')&(ptsEdgelat['direction']!='W')]

    ax.yaxis.set_ticks(ptsEdgelat.geometry.centroid.y)
    ax.yaxis.set_ticklabels(ptsEdgelat.display)

    #plt.show()

    ax.set_title('Industrial Area', pad=20)
    fig.savefig(f'{dirout}industrialArea_OSM-{continent}.png',dpi=400)
    plt.close(fig)
