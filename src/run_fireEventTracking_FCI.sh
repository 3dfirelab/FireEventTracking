#!/bin/bash
source ~/.myKeys.sh
if [ "$(hostname)" = "portmtg" ]; then
    export srcDir=/home/vost/Src/FireEventTracking/src
    export rootDataDir=/home/vost/Data
    export mambaDir=/home/vost/miniforge3/condabin/
elif [ "$(hostname)" = "pc70852" ]; then
    export srcDir=/home/paugam/Src/FireEventTracking/src
    export rootDataDir=/media/paugam/gast
    export mambaDir=/home/paugam/miniforge3/condabin/
elif [ "$(hostname)" = "andall" ]; then
    export srcDir=/home/paugam/Src/FireEventTracking/src
    export rootDataDir=/data/shared/
    export mambaDir=/home/paugam/miniforge3/condabin/
fi
#"$srcDir"/mount_aeris.sh

runName=$2

export dataDir=$rootDataDir/FCI/hotspots
export ctrlDir=$rootDataDir/FCI/log
export logDir=$rootDataDir/FCI/"$runName"_fire_events/log
if [ ! -d "$logDir" ]; then
    mkdir -p "$logDir"
fi


# --- Argument handling ---
if [ "$1" = "log_dir" ]; then
    echo "$logDir"
    exit 0

elif [ "$1" != "run" ]; then
    echo "Usage: $0 {log_dir|run} {runName}"
    exit 1
fi

echo "run FET"

LOCK_FILE=$ctrlDir/lock_FireEventTracking_$runName.txt
if [ ! -e $LOCK_FILE ]; then
    touch $LOCK_FILE

    $mambaDir/mamba run -n tracking python $srcDir/fireEventTracking.py --inputName $runName --sensorName FCI --log_dir $logDir #>& $logDir/fireEventTracking.log
    status_python=$?
    
    echo 'fireEventTracking.py done'


    if [ $status_python -ne 0 ]; then
        exit 1
    fi

    #rm "$ctrlDir/runFireEvent.txt"
    rm $LOCK_FILE

    #to concatenate last 2 days on the website
    #$mambaDir//mamba run -n tracking python $srcDir/fireEventTracking_updateWebSite.py
else
    echo "found $LOCK_FILE "
    exit 1
fi


