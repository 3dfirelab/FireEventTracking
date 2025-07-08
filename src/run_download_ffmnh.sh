#!/bin/bash
source ~/.myKeys.sh
export srcDir=/home/paugamr/Src/FireEventTracking/src
"$srcDir"/mount_aeris.sh

export ctrlDir=/home/paugamr/data/FCI/log

export logDir=/home/paugamr/data/FCI/fire_events/log
if [ ! -d "$logDir" ]; then
    mkdir -p "$logDir"
fi


# Call the Python script
if [ ! -e "$ctrlDir/lock_download_ffmnh.txt" ]; then
    touch "$ctrlDir/lock_download_ffmnh.txt"
    /home/paugamr/miniforge3/condabin/mamba run -n tracking python "$srcDir"/download_ffmnh.py  >& $logDir/download_ffmnh.log
    rm "$ctrlDir/lock_download_ffmnh.txt"
fi

