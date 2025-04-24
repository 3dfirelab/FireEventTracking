import subprocess
import glob
import requests
import os
from urllib.parse import urlparse
import sys

with open('./firmsToDownload.txt','r') as f :
    listurl = f.readlines()


for url in listurl:

    dirdata = '/mnt/data3/FireEventTracking/'

    # Extract filename from URL
    filename = os.path.basename(urlparse(url).path)

    if 'FIRE_SV' in filename:
        dirout = dirdata + 'VIIRS_SNPP_NRT/'
    elif 'FIRE_J1V' in filename: 
        dirout = dirdata + 'VIIRS_NOAA20_NRT/'
    elif 'FIRE_J2V' in filename: 
        dirout = dirdata + 'VIIRS_NOAA21_NRT/'
    else:
        print('did not recognize sat')
        sys.exit()

    filename = dirout + filename

    # Download and save
    response = requests.get(url)
    response.raise_for_status()

    with open(filename, "wb") as f:
        f.write(response.content)

    print(f"Downloaded and saved as {filename}")
