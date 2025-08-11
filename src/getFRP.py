import pandas as pd
import xarray as xr
import xarray as xr
import rioxarray
import geopandas as gpd
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt
import numpy as np 
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
import pdb 
import glob 
from datetime import datetime, timedelta, timezone
import os 
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable
import bisect
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.transforms import Bbox
import matplotlib.image as mpimg
from matplotlib import dates as mdates
from zoneinfo import ZoneInfo
from PIL import Image
from matplotlib.patches import Patch
import gc

#homebrewed
import fireEvent

if __name__ == '__main__':
    
    dirInFRP = '/media/paugam/gast/AERIS_2/FCI/fire_events/Pickles_active_2025-08-06_1000/'
    fireEvent_id = 1
    outputFile = 'ribaute_frp.csv'
    #
    # load FET data
    #
    event = fireEvent.load_fireEvent(f"{dirInFRP}/{fireEvent_id:09d}.pkl")

    event_times_series = pd.Series(event.times)
    frp_series = pd.Series(event.frps)

    df = pd.DataFrame({'timestamp': event_times_series, 'FRP': frp_series}).sort_values('timestamp')

    df.to_csv(outputFile)
