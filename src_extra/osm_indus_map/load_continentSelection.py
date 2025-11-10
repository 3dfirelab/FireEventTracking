

import sys
import os 
#os.environ['USE_PYGEOS'] = '0'
import pandas as pd 
import geopandas as gpd
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob
import shutil
import warnings
import importlib 
import pdb 

#homebrwed
import countries as contries_mod
sys.path.append('../src-map/')
import params
import tools

if __name__ == '__main__':
   
    importlib.reload(params)
    continent = 'med'
    dir_data = '/data/shared/'
    
    print('continent = ', continent)

    importlib.reload(contries_mod)
    from countries import europe, asia, namerica, samerica, camerica, africa, russia, oceania, easteurope, med
    #from countries_area_check import europeA, asiaA, namericaA, samericaA, camericaA, africaA, russiaA, oceaniaA, easteuropeA

    if continent == 'europe': 
        countries_selection = np.array(europe)

    elif continent == 'asia': 
        countries_selection = np.array(asia)
        countries_area = asiaA
    
    elif continent == 'namerica': 
        countries_selection = namerica
        countries_area = namericaA
    
    elif continent == 'samerica': 
        countries_selection = samerica
        countries_area = samericaA
    
    elif continent == 'camerica': 
        countries_selection = camerica
        countries_area = camericaA
    
    elif continent == 'africa': 
        countries_selection = africa
        countries_area = africaA
    
    elif continent == 'russia': 
        countries_selection = russia
        countries_area = russiaA
    
    elif continent == 'oceania': 
        countries_selection = oceania
        countries_area = oceaniaA
    
    elif continent == 'easteurope': 
        countries_selection = easteurope
        countries_area = easteuropeA
     
    elif continent == 'med': 
        countries_selection = med

    countries_selection = np.array(countries_selection)
    #countries_area = np.array(countries_area)

    params = params.load_param(continent)
    xminAll,xmaxAll = params['xminAll'], params['xmaxAll']
    yminAll,ymaxAll = params['yminAll'], params['ymaxAll']
    crs_here        = params['crs_here']
    bufferBorder    = params['bufferBorder']

    indir = '{:s}Boundaries/'.format(dir_data)
    bordersNE = gpd.read_file(indir+'NaturalEarth_10m_admin_0_countries/ne_10m_admin_0_countries.shp')
    
    continentMask = None
    for country_code,country_code2 in zip(countries_selection[:,1], countries_selection[:,2]): 
       
        if country_code is None: continue 

        if len(country_code.split(','))>1:
            country_code_ = country_code.split(',')
            condition = (bordersNE['SOV_A3']==country_code_[0])
            for country_code__ in country_code_[1:]:
                condition |= (bordersNE['SOV_A3']==country_code__)
            borders_ = bordersNE[condition]
        else:
            borders_ = bordersNE[bordersNE['SOV_A3']==country_code]

        if country_code2 is not None: 
            borders_ = borders_[bordersNE['ADM0_A3']==country_code2]

        #if country_code == 'ESP': 
        #    bounds_ = borders_.total_bounds
        #    bounds_[3] = 35.967
        #    borders_ = gpd.clip(borders_, bounds_)

        if continentMask is None:
            continentMask=borders_
        else:
            continentMask = pd.concat([continentMask,borders_])
            
    continentMask = continentMask.to_crs(crs_here)
    continentMask['LEVL_CODE'] = 0
    continentMask.to_file(indir+'mask_{:s}.geojson'.format(continent),driver='GeoJSON')
    
    if continentMask.crs.to_epsg() is None: 
        with open(indir+'mask_{:s}.prj'.format(continent),'w') as f:
            f.write(continentMask.crs.to_wkt())

    ax=plt.subplot(111)
    continentMask.plot(ax=ax)
    ax.set_xlim(xminAll,xmaxAll)
    ax.set_ylim(yminAll,ymaxAll)
    
    '''
    ax=plt.subplot(122)
    areaComparison = np.zeros([countries_area.shape[0],2])
    for ic, country_code in  enumerate(countries_area[:,1]):
        
        areaComparison[ic,0] = float(countries_area[ic,2])
        country_code_arr = country_code.split(',') 
        for country_code_ in country_code_arr:
            areaComparison[ic,1] += continentMask[continentMask['SOV_A3']==country_code_].area.sum()*1.e-6
    
    plt.scatter(areaComparison[:,0],areaComparison[:,1])
    plt.plot([0,areaComparison[:,0].max()],[0,areaComparison[:,0].max()],linestyle=':')
    '''
    plt.show()
    

