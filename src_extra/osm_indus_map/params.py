import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def load_param(continent):

    if continent == 'europe':
        xminAll,xmaxAll = 2500000., 7400000.
        yminAll,ymaxAll = 1400000., 5440568.
        crs_here = 'epsg:3035'
        bufferBorder = -1800
        distgroup = 5.e3
        lonlat_bounds = None # not necessary here, this is to plot land background 
        gratreso = 15

    elif continent == 'asia':
        xminAll,xmaxAll = -5.8e6, 5.5e6
        yminAll,ymaxAll = -3.6e6, 4.44e6
        crs_here = '+proj=aea +lon_0=-263.671875 +lat_1=-1.5736574 +lat_2=42.3499669 +lat_0=20.3881547 +datum=WGS84 +units=m +no_defs' #'epsg:8859'
        bufferBorder = -10000
        distgroup = 1.e3
        lonlat_bounds = [[0, -20, 170, 70.]] # no
        gratreso = 15
    
    elif continent == 'namerica':
        lonlat_bounds = [[-180,5,10,90],[150,60,180,90]]
        xminAll,xmaxAll = 1.95e6,  9.06e6
        yminAll,ymaxAll = -2.98e6,  5.90e6
        crs_here = 'epsg:3347'
        #xminAll,xmaxAll = -4.61e6,  2.62e6
        #yminAll,ymaxAll = -4.19e6,  4.18e6
        #crs_here = '+proj=laea +lon_0=-447.19 +lat_0=49.57 +datum=WGS84 +units=m +no_defs'
        #from https://projectionwizard.org/#
        #crs_here='ESRI:102008'
        #crs_here='+proj=aea +lat_0=40 +lon_0=-96 +lat_1=20 +lat_2=60 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs +type=crs'
        #xminAll,xmaxAll = -6859872.43, 4627500.55 
        #yminAll,ymaxAll = -1870790.97, 5421010.03

        bufferBorder = -10000
        distgroup = 1.e3
        gratreso = 15
    
    elif continent == 'samerica':
        #lonlat_bounds = [[-180,5,10,90],],
        xminAll,xmaxAll = -3.87e6,  3.03e6
        yminAll,ymaxAll = -3.02e6,  5.15e6
        #crs_here = 'epsg:31971'
        crs_here = 'ESRI:102015'
        bufferBorder = -5000
        distgroup = 1.e3
        lonlat_bounds = [[-105., -60.0, -20., 20.]] # not necessary here, this is to plot land background 
        gratreso = 15
    
    elif continent == 'camerica':
        #lonlat_bounds = [[-180,5,10,90],],
        xminAll,xmaxAll = -3.e5,  4.004e6
        yminAll,ymaxAll = 5.55e5,  3.15e6
        crs_here = 'epsg:31970'

        bufferBorder = -1000
        distgroup = 1.e3
        lonlat_bounds = [[-95, 0.0, -50, 50.]] # no
        gratreso = 5
    
    elif continent == 'africa':
        #lonlat_bounds = [[-180,5,10,90],],
        xminAll,xmaxAll = -4.47e6,  4.67e6
        yminAll,ymaxAll = -4.01e6,  4.23e6
        crs_here = 'ESRI:102011' #'+proj=chamb +lat_1=22 +lon_1=0 +lat_2=22 +lon_2=45 +lat_3=-22 +lon_3=22.5 +datum=WGS84 +type=crs'

        bufferBorder = -10000
        distgroup = 1.e3
        lonlat_bounds = [[-33, -40.0, 68, 40.]] # no
        gratreso = 15

    elif continent == 'russia':
        xminAll,xmaxAll = -4.85e6,  3.97e6
        yminAll,ymaxAll = -1.57e6,  3.83e6
        crs_here = '+proj=aea +lat_1=50 +lat_2=70 +lat_0=56 +lon_0=100 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs ' #'+proj=chamb +lat_1=22 +lon_1=0 +lat_2=22 +lon_2=45 +lat_3=-22 +lon_3=22.5 +datum=WGS84 +type=crs'

        bufferBorder = -100
        distgroup = 1.e3
        lonlat_bounds = [[-5, 0, 180, 90.],[-180,40,-150,90]] # no
        gratreso = 5
    
    elif continent == 'oceania':
        xminAll,xmaxAll = -7.3e6, 3.8e6
        yminAll,ymaxAll = -4.2e6,  3.8e6
        crs_here = '+proj=aea +lon_0=-171.5625 +lat_1=-39.4044996 +lat_2=-2.4319777 +lat_0=-20.9182387 +datum=WGS84 +units=m +no_defs' #'EPSG:3832' 

        bufferBorder = -10000
        distgroup = 1.e3
        lonlat_bounds = [[20, -65, 180, 30.],[-180,-65,-120,30]] # no
        gratreso = 15
    
    elif continent == 'easteurope':
        #lonlat_bounds = [[-180,5,10,90],],
        xminAll,xmaxAll = 3.05e5, 2.52e6
        yminAll,ymaxAll = 4.67e6,  6.307e6
        crs_here = 'epsg:6381'

        bufferBorder = -1000
        distgroup = 1.e3
        lonlat_bounds = [[18., 35. , 55., 60.]] # no
        gratreso = 5

    params = {'xminAll': xminAll,
              'xmaxAll': xmaxAll, 
              'yminAll': yminAll,
              'ymaxAll': ymaxAll, 
              'crs_here': crs_here, 
              'bufferBorder': bufferBorder, 
              'distgroup': distgroup, 
              'lonlat_bounds':lonlat_bounds,
              'gratreso':gratreso
              }
    

    return params
