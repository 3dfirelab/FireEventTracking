import pandas as pd
import glob 
import sys
import zipfile
import os
import shutil
import pdb 

#data_dir = '/mnt/data3/FireEventTracking/'
data_dir = '/home/paugam/Data/2025_ofunato/'

satnames = ['VIIRS_SNPP_NRT', 'VIIRS_NOAA20_NRT', 'VIIRS_NOAA21_NRT']

for satname in satnames: 
    dirIn = '{:s}/{:s}/'.format(data_dir,satname)
    filesIn = glob.glob(dirIn+'*.zip')

    for zip_path in filesIn:
        zip_dir = os.path.dirname(zip_path)
        base_name = os.path.splitext(os.path.basename(zip_path))[0]
        
        # Path to OLD folder
        old_dir = os.path.join(zip_dir, 'OLD')
        os.makedirs(old_dir, exist_ok=True)  # Create OLD directory if it doesn't exist
        
        # Extract contents
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(zip_dir)
        
        filesInCSV = glob.glob(zip_dir+'/*.csv')  
        for fileInCSV in filesInCSV:
            # Load the CSV file
            df = pd.read_csv(fileInCSV)
            
            year = pd.to_datetime(df['acq_date'][0]).year
            dirout = dirIn #'{:s}{:4d}/'.format(dirIn,year)
            os.makedirs(dirout,exist_ok=True)

            # Ensure acq_time is zero-padded to 4 digits
            df['acq_time'] = df['acq_time'].apply(lambda x: f"{int(x):04d}")

            # Extract the hour part and zero out minutes
            df['acq_hour'] = df['acq_time'].str[:2] + '00'

            # Group by acquisition date and save each group as a separate CSV
            for (date, hour), group in df.groupby(['acq_date', 'acq_hour']):
            #for date, group in df.groupby('acq_date'):
                filename = '{:s}/{:s}_{:s}-{:s}.csv'.format(dirout,satname.lower(),date,hour)
                group.to_csv(filename, index=False)
                print(f'Saved {filename} with {len(group)} records.')

        
        # Move extracted folder and zip file to OLD
        shutil.move(zip_path, os.path.join(old_dir, os.path.basename(zip_path)))
        [shutil.move(xx, old_dir) for xx in filesInCSV]

