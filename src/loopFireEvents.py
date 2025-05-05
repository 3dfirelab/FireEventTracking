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
from shapely.geometry import Point, MultiPoint
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

#home brewed
import fireEvent
import fireEventTracking


#############################
if __name__ == '__main__':
#############################
    importlib.reload(fireEvent)

    #init
    params = fireEventTracking.init('AVEIRO') 
    currentTime = datetime.strptime('2024-09-20_2100', '%Y-%m-%d_%H%M')
    
    #loop
    fireEvent.loopFireEvents(params, currentTime )



