#!/bin/bash

export srcDir=/home/paugamr/Src/FireEventTracking/src
"$srcDir"/mount_aeris.sh

export dataDir=/home/paugamr/data/VIIRS/
export logDir=/home/paugamr/data/VIIRS/fire_events/log
if [ ! -d "$logDir" ]; then
    mkdir -p "$logDir"
fi


# Get current time in desired format: 2025-06-14T22:00:00
current_time=$(date -u +"%Y-%m-%dT%H:%M:%S")

# Call the Python script with the timestamp as argument
/home/paugamr/miniforge3/condabin/mamba run -n tracking python "$srcDir"/rasterizeFRP_MOCAGE.py "$current_time" 12 "$dataDir"  >& $logDir/rasterize.log

#if grep -qs "[[:space:]]$aerisDir[[:space:]]" /proc/self/mountinfo; then
#    umount $aerisDir
#fi

