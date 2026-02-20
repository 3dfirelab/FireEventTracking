#!/bin/bash
#
#SBATCH --job-name=FETStat # updated below in the next submission
#SBATCH --ntasks=22
#SBATCH --cpus-per-task=1
#SBATCH -p prod
#SBATCH --mem 30G
#SBATCH --time=06:00:00
#
start_time=$(date +%s)

if [ -z "$1" ]; then
    echo "ERROR: runName not provided (argument 1 is missing)" >&2
    exit 1
fi
runName=$1
cd $SLURM_SUBMIT_DIR

source ~/.myKeys.sh
source ~/miniforge3/bin/activate tracking
python  fireEventStats.py --inputName $runName --sensorName FCI

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
h=$(( elapsed / 3600 ))
m=$(( (elapsed % 3600) / 60 ))
s=$(( elapsed % 60 ))

echo "Elapsed time: ${h}h ${m}m ${s}s"

