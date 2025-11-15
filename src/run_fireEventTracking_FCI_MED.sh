#!/bin/bash
source ~/.myKeys.sh
if [ "$(hostname)" = "portmtg" ]; then
    export srcDir=/home/vost/Src/FireEventTracking/src
    export rootDataDir=/home/vost/Data
    export mambaDir=/home/vost/miniforge3/condabin/
elif [ "$(hostname)" = "pc70852" ]; then
    export srcDir=/home/paugam/Src/FireEventTracking/src
    export rootDataDir=/mnt/media/paugam/gast/AERIS_2
    export mambaDir=/home/paugam/miniforge3/condabin/
elif [ "$(hostname)" = "andall" ]; then
    export srcDir=/home/paugam/Src/FireEventTracking/src
    export rootDataDir=/data/shared/
    export mambaDir=/home/paugam/miniforge3/condabin/
fi
#"$srcDir"/mount_aeris.sh

export dataDir=$rootDataDir/FCI/hotspots
export ctrlDir=$rootDataDir/FCI/log
export logDir=$rootDataDir/FCI/MED_fire_events/log
if [ ! -d "$logDir" ]; then
    mkdir -p "$logDir"
fi

#if [ -f "$ctrlDir/runFireEvent.txt" ] && [ ! -e "$ctrlDir/lock_FireEventTracking.txt" ]; then
if [ ! -e "$ctrlDir/lock_FireEventTracking.txt" ]; then
    touch "$ctrlDir/lock_FireEventTracking.txt"

    $mambaDir/mamba run -n tracking python $srcDir/fireEventTracking.py --inputName MED --sensorName FCI --log_dir $logDir >& $logDir/fireEventTracking.log

    #rm "$ctrlDir/runFireEvent.txt"
    rm "$ctrlDir/lock_FireEventTracking.txt"

    #to concatenate last 2 days on the website
    #$mambaDir//mamba run -n tracking python $srcDir/fireEventTracking_updateWebSite.py
fi


