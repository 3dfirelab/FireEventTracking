import sys
import os 
#os.environ['USE_PYGEOS'] = '0'
import pandas as pd 
import geopandas as gpd
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
import pyrosm
import glob
import shutil
import warnings
import importlib 
import pdb 

#homebrwed
import countries as contries_mod
sys.path.append('../src-map/')
import tools

if __name__ == '__main__':
   
    continent = 'europe'
    dir_data = tools.get_dirData()
    dir_out_root = '/mnt/dataEstrella2/OSM_IndustrialZone/'
    '''
    importlib.reload(contries_mod)
    from countries import europe, asia, namerica, samerica, camerica, africa
    
    if continent == 'europe': 
        countries_selection = europe
    elif continent == 'asia': 
        countries_selection = asia
    elif continent == 'namerica': 
        countries_selection = namerica
    elif continent == 'samerica': 
        countries_selection = samerica
    elif continent == 'camerica': 
        countries_selection = camerica
    elif continent == 'africa': 
        countries_selection = africa
    '''
    warnings.filterwarnings("ignore")

    flag_industrial     = True
    flag_veg_osm        = False
    
    indir = '{:s}OSM/PerCountry-{:s}/'.format(dir_data,continent)
    osmfiles = sorted(glob.glob(indir+'*osm.pbf'))
  
    if flag_industrial:
        print('industrial')
    elif flag_veg_osm:
        print('veg-osm')

    for osmfile in osmfiles: 
        country =os.path.basename(osmfile).split('-latest')[0]
       
        #if country == 'netherlands': sys.exit()
        fileintmp = glob.glob('/tmp/*.osm.pbf')
        if len(fileintmp)>1:
            for file_ in fileintmp:
                os.remove(file_)
        osmfile_='/tmp/'+os.path.basename(osmfile)
        shutil.copyfile(osmfile, osmfile_ )

        if flag_industrial:
             
            outdir = '{:s}/{:s}/'.format(dir_out_root, continent)
            tools.ensure_dir(outdir)

            nbreTile = len( glob.glob( osmfile.split('-latest')[0]+'*.osm.pbf') )
            if nbreTile > 1: 
                it = int(osmfile.split('-latest_')[1].split('.osm')[0])
            else:
                it = None

            if it is None: 
                if os.path.isfile(outdir+'industrial-{:s}.geojson'.format(country)): continue
                if os.path.isfile(outdir+'industrial-{:s}_dist.geojson'.format(country)): continue
            else:
                if os.path.isfile(outdir+'industrial-{:s}-{:d}.geojson'.format(country,it)): continue
                if os.path.isfile(outdir+'industrial-{:s}-{:d}_dist.geojson'.format(country,it)): continue
            
            if it is None:
                print('  {:s}  ...'.format(country), end='\r')
            else:
                print('  {:s} {:d}/{:d} ... {:s}'.format(country,it,nbreTile, os.path.basename(osmfile_)), end='\r')
            sys.stdout.flush()

            try:
                osm = pyrosm.OSM(filepath=osmfile_)
                
                custom_filter={'landuse': ['industrial', 'landfill', 'quarry', 'military']}
                try: 
                    indus = osm.get_data_by_custom_criteria(custom_filter=custom_filter)
                except (ValueError, KeyError) as mm:
                    indus = None
            except: 
                    print('  {:s} {:s}  could not open osm file    '.format(country, os.path.basename(osmfile_)) )
                    indus = None
                    continue

            if indus is None: 
                print('  {:s} {:s}  no data    '.format(country, os.path.basename(osmfile_)) )
                continue
            indus = indus[indus.geom_type!='MultiPoint'] 
            indus = indus[indus.geom_type!='Point'] 
            indus = indus[indus.geom_type!='MultiLineString']
            indus = indus[indus.geom_type!='LineString']
            
            if  it is None:
                indus.to_file(outdir+'industrial-{:s}.geojson'.format(country), driver="GeoJSON")
            else: 
                indus.to_file(outdir+'industrial-{:s}-{:d}.geojson'.format(country,it), driver="GeoJSON")

        print('  {:s}  done                                       '.format(country) )



        categories = np.arange(1,6)
        colorveg=['blue','green','orange','black','magenta']

        if flag_veg_osm:
            print('fuel osm')
            outdir = '{:s}FuelCategories-OSM/'.format(dir_data)
            
            extent_ll=[-23.73507226782506, 24.505457173625324, 99.52727619009086, 80.51193780175987]
            osm = pyrosm.OSM(filepath=osmfile_, bounding_box=extent_ll)

            custom_filter_all = []
            custom_filter_all.append( [{'landuse':['forest'], 'leaf_type':['needleleaved']}, 
                                       {'natural':['wood'],   'leaf_type':['needleleaved']}
                                      ])
            custom_filter_all.append( [{'landuse':['forest'],'leaf_type':['mixed']    }, 
                                       {'natural':['wood'],  'leaf_type':['mixed']    }
                                      ])
            custom_filter_all.append( [{'landuse':['forest'], 'leaf_type':['brodleaved']}, 
                                       {'natural':['wood'],   'leaf_type':['brodleaved']},
                                       {'natural':['scrub']},
                                       {'natural':['meadow']},
                                       {'natural':['grassland']},
                                      ])
            custom_filter_all.append( [{'natural':['heath']}] )
            custom_filter_all.append( [{'natural':['wetlands']}] )

            #ax = plt.subplot(111)
            for iv in categories:
                print('  fuel cat: {:d}'.format(iv))
              
                flag_init_cat = True
                for custom_filter in custom_filter_all[iv-1]:
                    flag_init_element = True
                    for ik, key in enumerate(custom_filter.keys()):
                        print('    filter: {:s} - {:s}'.format(key, custom_filter[key][0]))

                        
                        element_ = osm.get_data_by_custom_criteria(custom_filter={key:custom_filter[key]})
                        if element_ is None: 
                            element = None
                            break
                        element_ = element_[element_.geom_type!='MultiPoint'] 
                        element_ = element_[element_.geom_type!='Point'] 
                        element_ = element_[element_.geom_type!='MultiLineString']
                        element_ = element_[element_.geom_type!='LineString']
                        element_ = element_.to_crs(23029) 
                        
                        if flag_init_element:
                            element = element_
                            flag_init_element = False
                        else: 
                            #mm = gpd.overlay(element, element_, how = 'intersection')
                            element = gpd.overlay(element, element_, how = 'intersection')
                            #element = gpd.sjoin(element, element_, how='left', predicate='overlaps')
                    #keys = custom_filter.keys()
                    #condition = (pd.Series([True]*len(element)))
                    #for key in keys: 
                    #    condition &= (element[key]==custom_filter[key][0]) 
                    #element = element[condition]

                    #element = element[(element['landuse']=='forest')&\
                    #                  (element['natural']=='wood')&\
                    #                  (element['leaf_type']=='needleleaved')]
                    
                    print(' ')
                    if element is None: continue
                    if flag_init_cat:
                        fuel = element 
                        flag_init_cat = False
                    else:
                        fuel = gpd.overlay(fuel, element, how = 'union')

                fuel = fuel.drop_duplicates('geometry')

                fuel.to_file(outdir+'fuelCategory{:d}-{:s}.shp'.format(iv,country))
                #fuel.plot(ax=ax,color=colorveg[iv-1], alpha=.5)
        
            #plt.show()
        '''
        if flag_forest:
            outdir = '/mnt/dataEstrella/OSM/FuelCategories/'
            custom_filter={'landuse': ['forest']}
            forest = osm.get_landuse(custom_filter=custom_filter)
            forest = forest[forest.geom_type!='MultiPoint'] 
            forest = forest[forest.geom_type!='Point'] 
            forest = forest[forest.geom_type!='MultiLineString']
            forest = forest[forest.geom_type!='LineString']

            forest.to_file(outdir+'forest.shp')
        '''
        
        os.remove(osmfile_)

