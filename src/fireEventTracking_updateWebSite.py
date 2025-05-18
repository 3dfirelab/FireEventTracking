import glob
import os
import re
from datetime import datetime, timedelta
import shutil
import pdb 
import json
import requests
import geopandas as gpd 
import pandas as pd

# Define source and destination
source_dir = '/mnt/dataEstrella2/SILEX/VIIRS-HotSpot/FireEvents/'
dest_file = '/mnt/homeEstrella/WebSite/leaflet/data/firEvents_merged_last2days.geojson'

#SELECT LAST FILES over the last 2 days
# Define pattern to extract datetime from filename
pattern = re.compile(r'firEvents-(\d{4}-\d{2}-\d{2}_\d{4})\.geojson')

# Define the source directory and collect matching files
geojson_files = glob.glob(os.path.join(source_dir, 'GeoJson', 'firEvents-*.geojson'))

# Time window
now = datetime.utcnow()
seven_days_ago = now - timedelta(days=2)

# List to store matching files
recent_files = []

# Loop through and filter files from the last 2 days
for f in geojson_files:
    match = pattern.search(os.path.basename(f))
    if match:
        dt_str = match.group(1)
        dt = datetime.strptime(dt_str, '%Y-%m-%d_%H%M')
        if seven_days_ago <= dt <= now:
            recent_files.append(f)

# Load and concatenate GeoDataFrames
gdfs = [gpd.read_file(f) for f in recent_files]
if gdfs:
    merged_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
    
    # Save to one output file
    output_path_merged = './firEvents_merged_last2days.geojson'
    merged_gdf.to_file(output_path_merged, driver='GeoJSON')


#SELECT LAST FILE
# Pattern to match timestamp in filename
pattern = re.compile(r'firEvents-(\d{4}-\d{2}-\d{2}_\d{4})\.geojson')

# Find all matching files
geojson_files = glob.glob(source_dir+'/GeoJson/firEvents-*.geojson')

# Extract datetimes and find the latest one
latest_file = None
latest_dt = None

for f in geojson_files:
    match = pattern.search(os.path.basename(f))
    if match:
        dt_str = match.group(1)
        dt = datetime.strptime(dt_str, '%Y-%m-%d_%H%M')
        if latest_dt is None or dt > latest_dt:
            latest_dt = dt
            latest_file = f

# Move the latest file to the destination
if gdfs:
    shutil.move(output_path_merged, dest_file)
    shutil.copy2(source_dir+ 'log/fireEventTracking.log', '/mnt/homeEstrella/WebSite/leaflet/data/logs/fireEventTracking.log')
    print(f"Moved latest file: {output_path_merged} → {dest_file}")
    
    #and push to corte
    try: 
        print('--') 
        url = "https://forefire.univ-corse.fr/live/getRonan.php"
        headers = {
                    "Content-Type": "application/json"
                    }
        with open(latest_file, "r") as f:
                geojson_dict = json.load(f)
        response = requests.post(url, headers=headers, data=json.dumps(geojson_dict))

        print("Status code:", response.status_code)
        print("Response:", response.text)
        print('--') 
    except: 
        pass
else:
    print("No matching files found.")
