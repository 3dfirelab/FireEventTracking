# FET -- Fire Event Tracking
Compute spatio-temporal clustering of hotspots to create fire Event.
The executable is `src/run_fireEventTracking.sh`.

It was developed  with FCI and VIIRS  hostsport product, though in the lastest version FCI was mostly tested.

with FCI, FET run at the 10 min cadence of the system. Saving is done at a different frequency set to 30 min in the current version.

Hotspot are clustered using the following rules:
- the spatial clustering is done with DBSCAN using `800`m (2x VIIRS resolution) as a max distance between sample. 
```
        db = DBSCAN(eps=epsilon, min_samples=1, metric='euclidean').fit(np.array(hsgdf_all[['x','y' ]]))
```
- 31 days life time for each hotspot if nor removed by end of event.
- event is considered off if not update within 2 days.

Each event has a unique `id={:09d}.format()`. Info for each event are stored in Pickle format link to the class `fireEvent`. Every `30 min` of a run, info of the active event are stored in `Pickles_active_{current_date}/`. 
At every 30 min, all active fire event are also saved for the whole domain in `GeoJson/firEvents-{current_date}.geojson`. (this could be removed for operational run, it helped for the dev)

## Example of set up with FCI:
We present here an example to run on the Ribaute fire that took place in Southern France on the 5th of August 2025.

### configuration file
In the dir `config` there is a configuration file for this example.
```
# config.yaml
general:
  domain: 2.2,42.7,3.2,43.5
  domainName: 'RIBAUTE'
  crs: 3035
  use_sedoo_drive: False 
  root_data: /data/paugam/FET-TEST
hs:
  dir_data: ORIGIN/hotspots2025
  sats: ['']
event:
  dir_data: ORIGIN/RIBAUTE_fire_events 
  dir_geoJson: ORIGIN/RIBAUTE_fire_events/GeoJson
  dir_frp: ORIGIN/RIBAUTE_fire_events/FRP
  dir_cloudMask: ORIGIN/RASTER/cloudMask/
  start_time: '2025-08-05_1400'
  end_time: '2025-08-06_1200'
  mask_HS: /data/paugam/FET-TEST/HS-MASK/data_mask_buffer_3857.geojson
  osm_bndf: 'OSM_Boundaries/osm_MED_admin_muni_boundaries.gpkg'
  cloudMask_mtg: "FCI/RASTER/cloudMask/"
  dir_hs_log: "ORIGIN/RIBAUTE_fire_events/HS_log/"
postproc:
  dir_FWI:  AROME/FWI
  dir_WIND: AROME/FORECAST/WIND/
```
the `hs` section is used by another code that download the hotspot.
As well as the postproc section. 
`ORIGIN` is replace by the sensor name that is defined in the script that control the execution. you have one execution file per sensor (FCI or VIIRS).

`start_time` and  `end_time` control the start and end of the run. 
if `end_time_hard` is specified instead of `end_time`, the code is set to run by increment of 24h until it reaches `end_time_hard`. 
The last time os the run is stored in the `log` directory of the run and can be used as the restart at the next execution.


### data tree directory
The data directory tree is located at: `root_data`+`ORIGIN`
(`ORIGIN` = `FCI` | `VIIRS`)
The directory has the followin structure (example below has only one fire event):
```
hotspots
├── LSA-509_MTG_MTFRPPIXEL-ListProduct_MTG-FD_202507260000.csv
...
└── LSA-509_MTG_MTFRPPIXEL-ListProduct_MTG-FD_202507282350.csv
RIBAUTE_fire_events/
├── FRP-FROS
│   └── 000000000.json
├── GeoJson
│   ├── firEvents-2025-08-05_1400.geojson
...
│   └── firEvents-2025-08-06_1100.geojson
├── HS_log
│   ├── HS-2025-08-05_1400.csv
...
│   └── HS-2025-08-06_1200.csv
├── Pickles_active_2025-08-05_1400
│   └── 000000000.pkl
...
├── Pickles_active_2025-08-06_1100
│   └── 000000000.pkl
├── hotspots-2025-08-05_1400.gpkg
...
├── hotspots-2025-08-06_1100.gpkg
└── log
    └── timeControl_RIBAUTE.txt
log/
└── lock_FireEventTracking_RIBAUTE.txt
```
- `RIBAUTE_fire_events/log` store the time of the end od the run, to use for restart.
- `RIBAUTE_fire_events/FRP-FROS` store one json file per fire event with te timeseries o the FRP and FROS.
- `log` stored which run is active to avoid overwritting.
- `hotspots*.gpkg` stored the hostpot at every 30 min saving time interval.
- `fix=hotspots*.gpkg` stored the hostpot detected in the fix hotspot mask at every 30 min saving time interval.
- in `HS_log`, stats on the hotspot and fixhotsport is stored.

### runing FET
Before a new run the `log` directory that contained the lock file need to exsit.
and in `run_fireEventTracking_FCI.sh` the two directory below need to be defined:
```
    export srcDir=/home/paugam/Src/FireEventTracking/src
    export mambaDir=/home/paugam/miniforge3/condabin/
```
then a simple run like for the Ribaute case is done with : 
```
./run_fireEventTracking_FCI.sh "run" RIBAUTE
```

there is also available a slurm command that can sequentially call `run_fireEventTracking_FCI` until it reach the `end_time_hard`. 
or will never stop the sequence if neither `end_time` or `end_time_hard` are specidified.


