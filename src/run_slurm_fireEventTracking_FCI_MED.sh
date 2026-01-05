#!/bin/bash
#
#   #SBATCH --account=paugam
#SBATCH --job-name=FET # updated below in the next submission
#SBATCH --ntasks=22
#SBATCH --cpus-per-task=1
#SBATCH -p prod
#SBATCH --mem 30G
#SBATCH --time=06:00:00
#

start_time=$(date +%s)

cd $SLURM_SUBMIT_DIR

export log_dir=`$SLURM_SUBMIT_DIR/run_fireEventTracking_FCI_MED.sh "log_dir"`
echo $log_dir

LOCK_FILE="$log_dir/../../log/lock_FireEventTracking.txt"

if [[ -e "$LOCK_FILE" ]]; then
    echo "found lock file: $LOCK_FILE"
    exit 1
fi

# extract YYYY-MM-DD from first field, then get month and day
tc_file="$log_dir/timeControl.txt"
ymd=$(head -n 1 "$tc_file" | awk -F'[_ ]' '{print $1}')     # 2025-02-24

if [ -z "$ymd" ]; then
    jobname="FET_xxxx"
else
    ymd=$(date -d "$ymd +1 day" +%Y-%m-%d)
    month=${ymd:5:2}
    day=${ymd:8:2}
    jobname="FET_${month}${day}"
fi

if [ ! -e "$log_dir/reach_end_time_hard.txt" ]; then
    # submit next job, but it will start only after this one ends
    sbatch --dependency=afterok:$SLURM_JOB_ID --job-name="$jobname" "$SLURM_SUBMIT_DIR/run_slurm_fireEventTracking_FCI_MED.sh"
fi

$SLURM_SUBMIT_DIR/run_fireEventTracking_FCI_MED.sh "run"
status_fet=$?

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
h=$(( elapsed / 3600 ))
m=$(( (elapsed % 3600) / 60 ))
s=$(( elapsed % 60 ))

echo "Elapsed time: ${h}h ${m}m ${s}s"

if [ $status_fet -ne 0 ]; then
    exit 1
fi

