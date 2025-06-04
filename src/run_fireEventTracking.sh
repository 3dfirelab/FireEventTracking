#!/bin/bash
#
source $HOME/miniconda3/bin/activate tracking
export dataDir=/mnt/dataEstrella2/SILEX/VIIRS-HotSpot
export srcDir=/home/paugam/Src/FireEventTracking/src
export logDir=/mnt/dataEstrella2/SILEX/VIIRS-HotSpot/FireEvents/log
if [ ! -d "$logDir" ]; then
    mkdir -p "$logDir"
fi

if [ -f "$dataDir/runFireEvent.txt" ] && [ ! -e "$dataDir/lock_FireEventTracking.txt" ]; then
    touch "$dataDir/lock_FireEventTracking.txt"
    python $srcDir/fireEventTracking.py --inputName SILEX --log_dir $logDir >& $logDir/fireEventTracking.log
    rm "$dataDir/runFireEvent.txt"
    rm "$dataDir/lock_FireEventTracking.txt"
    python $srcDir/fireEventTracking_updateWebSite.py 
fi
