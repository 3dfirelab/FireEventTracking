#!/bin/bash
source ~/.myKeys.sh
export srcDir=/home/paugamr/Src/FireEventTracking/src
"$srcDir"/mount_aeris.sh

export dataDir=/home/paugamr/data/VIIRS/hotspots
export ctrlDir=/home/paugamr/data/VIIRS/log
export logDir=/home/paugamr/data/VIIRS/fire_events/log
if [ ! -d "$logDir" ]; then
    mkdir -p "$logDir"
fi

if [ -f "$ctrlDir/runFireEvent.txt" ] && [ ! -e "$ctrlDir/lock_FireEventTracking.txt" ]; then
    echo "run fireEventTracking.py"
    touch "$ctrlDir/lock_FireEventTracking.txt"

    /home/paugamr/miniforge3/condabin/mamba run -n tracking python $srcDir/fireEventTracking.py --inputName SILEX-MF --sensorName VIIRS --log_dir $logDir >& $logDir/fireEventTracking.log

    rm "$ctrlDir/runFireEvent.txt"
    rm "$ctrlDir/lock_FireEventTracking.txt"

    #to concatenate last 2 days on the website
    #/home/paugamr/miniforge3/condabin/mamba run -n tracking python $srcDir/fireEventTracking_updateWebSite.py
fi

#if grep -qs "[[:space:]]$aerisDir[[:space:]]" /proc/self/mountinfo; then
#    umount $aerisDir 
#fi

