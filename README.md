# FET -- Fire Event Tracking
Compute spatial and temporal cluster of VIIRS hotspots to create vector fire Event.
The executable is `src/run_fireEventTracking.sh`.
Tis is run every time there are new hotsport available. See the repository [hsVIIRS](https://github.com/3dfirelab/hsVIIRS) to donwlaod VIIRS hotspots in NRT. 
hsVIIRS and FET share the same congif file format, see `config/config-SILEX.yaml`:
````
# config.yaml
general:
  domain: -10,35,20,52
  domainName: 'SILEX'
  crs: 25829
hs:
  dir_data: /mnt/dataEstrella2/SILEX/VIIRS-HotSpot
  sats: ['VIIRS_NOAA20_NRT', 'VIIRS_NOAA21_NRT', 'VIIRS_SNPP_NRT']
event:
  dir_data: /mnt/dataEstrella2/SILEX/VIIRS-HotSpot/FireEvents
  dir_geoJson: /mnt/dataEstrella2/SILEX/VIIRS-HotSpot/FireEvents/GeoJson
  start_time: '2025-05-01_0000'
```
hsVIIRS creates `config['hs']['dir_data']`:
```
.
├── log
├── VIIRS_NOAA20_NRT
├── VIIRS_NOAA21_NRT
└── VIIRS_SNPP_NRT
```

FET generates the fire event tracking data in `config['event']['dir_data']`
```
.
├── Fig/
├── firEvents-2025-05-01_0400.gpkg
├── .... 
── GeoJson/
│   ├── firEvents-2025-05-01_0400.geojson
│   ├── ... 
├── hotspots-2025-05-01_0400.gpkg
├── ... 
├── log/
│   ├── cron.log
│   ├── fireEventTracking.log
│   └── timeControl.txt
├── Pickles_active_2025-05-01_0400/
├── ... 
└── Pickles_past/
    ├── 000002687_2025-05-04_0052.pkl
    ├── ....
```


# ROS
Calculation of ROS is available in `src/loopFireEvents.py`.
this is still in dev.

### requirement
```
sudo apt install gdal-bin
sudo apt install texlive-latex-base texlive-latex-extra
sudo apt install cm-super
```
