import glob
import os
import re
from datetime import datetime
import shutil
import pdb 

# Define source and destination
source_dir = '/mnt/dataEstrella2/SILEX/VIIRS-HotSpot/FireEvents/GeoJson/'
dest_file = '/mnt/homeEstrella/WebSite/leaflet/data/firEvents-latest.geojson'

# Pattern to match timestamp in filename
pattern = re.compile(r'firEvents-(\d{4}-\d{2}-\d{2}_\d{4})\.geojson')

# Find all matching files
geojson_files = glob.glob(os.path.join(source_dir, 'firEvents-*.geojson'))

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
if latest_file:
    shutil.copy2(latest_file, dest_file)
    print(f"Moved latest file: {latest_file} → {dest_file}")
else:
    print("No matching files found.")
