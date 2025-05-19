# FET -- Fire Event Tracking
Compute spatial and temporal cluster of VIIRS hotspots to create vector fire Event.
The executable is `src/run_fireEventTracking.sh`.
Tis is run every time there are new hotsport available. See the repository [hsVIIRS](https://github.com/3dfirelab/hsVIIRS) to donwlaod VIIRS hotspots in NRT. 
hsVIIRS and FET share the same congif file format, see `config/config-SILEX.yaml`:
```
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
`start_time` is important to set for the first run. After it always start from the end of the last run.
hsVIIRS creates `config['hs']['dir_data']`:
```
.
├── log
├── VIIRS_NOAA20_NRT
├── VIIRS_NOAA21_NRT
└── VIIRS_SNPP_NRT
```
FET generates the fire event tracking data in `config['event']['dir_data']`. 
```
.
├── Fig/
├── GeoJson/
│   ├── firEvents-2025-05-01_0400.geojson
│   ├── ... 
├── hotspots-2025-05-01_0400.gpkg
├── ... 
├── log/
│   ├── cron.log
│   ├── fireEventTracking.log
│   └── timeControl.txt
├── Pickles_active_2025-05-01_0400/
Pickles_active_2025-05-01_0400/
│   ├── 000002687.pkl
|   ├── ... 
|   └── 000002766.pkl
└── Pickles_past/
    ├── 000002687_2025-05-04_0052.pkl
    └── ... 
```
Hotspot are clustered using the following rules:
- the spatial clustering is done with DBSCAN using `800`m (2x VIIRS resolution) as a max distance between sample. 
```
        db = DBSCAN(eps=epsilon, min_samples=1, metric='euclidean').fit(np.array(hsgdf_all[['x','y' ]]))
```
- 7 days life time for each hotspot
- event is considered off if not update within 2 days.

Each event has a unique `id={:09d}.format()`. Info for each event are stored in Pickle format. At every hour of a run, info of the active event are stored in `Pickles_active_{current_date}/`. One off the pickle is move to the directory `Pickles_past`. At every hour, all active fire event are also saved for the whole domain in `GeoJson/firEvents-{current_date}.geojson`.

`Fig/` directory stores png images made every day at 20h00 for the last days only, no history is kept.

Before Setting up into a cron using:
```
5 * * * * /home/paugam/Src/FireEventTracking/src/run_fireEventTracking.sh &> /mnt/dataEstrella2/SILEX/VIIRS-HotSpot/FireEvents/log/cron.log
```
you need to create the dir `/mnt/dataEstrella2/SILEX/VIIRS-HotSpot/FireEvents/log/`.
FET is run 5min after every hour. hsVIIRS is run at 00h of every hour, and expected to be completed if data are available.

# ROS
Calculation of ROS is available in `src/loopFireEvents.py`.
this is still in dev.

### requirement
```
sudo apt install gdal-bin
sudo apt install texlive-latex-base texlive-latex-extra
sudo apt install cm-super
```
