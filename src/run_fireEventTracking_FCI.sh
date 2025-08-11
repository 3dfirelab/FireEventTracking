#!/bin/bash
source ~/.myKeys.sh
export srcDir=/home/paugam/Src/FireEventTracking/src
#"$srcDir"/mount_aeris.sh

export dataDir=/media/paugam/gast/AERIS_2/FCI/hotspots
export ctrlDir=/media/paugam/gast/AERIS_2/FCI/log
export logDir=/media/paugam/gast/AERIS_2/FCI/fire_events/log
if [ ! -d "$logDir" ]; then
    mkdir -p "$logDir"
fi

#if [ -f "$ctrlDir/runFireEvent.txt" ] && [ ! -e "$ctrlDir/lock_FireEventTracking.txt" ]; then
if [ ! -e "$ctrlDir/lock_FireEventTracking.txt" ]; then
    touch "$ctrlDir/lock_FireEventTracking.txt"

    /home/paugam/miniforge3/condabin/mamba run -n tracking python $srcDir/fireEventTracking.py --inputName ribaute --sensorName FCI --log_dir $logDir >& $logDir/fireEventTracking.log

    #rm "$ctrlDir/runFireEvent.txt"
    rm "$ctrlDir/lock_FireEventTracking.txt"

    #to concatenate last 2 days on the website
    #/home/paugamr/miniforge3/condabin/mamba run -n tracking python $srcDir/fireEventTracking_updateWebSite.py
fi

#if grep -qs "[[:space:]]$MOUNT_POINT[[:space:]]" /proc/self/mountinfo; then
#    umount $MOUNT_POINT 
#fi

