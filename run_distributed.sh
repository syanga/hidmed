#!/bin/bash
############################## Submit Job ######################################
#SBATCH --time=24:00:00
#SBATCH --job-name="hidmed"
#SBATCH --output=hidmed_%j.txt
#SBATCH --error=hidmed_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --partition=normal
#####################################

# run using sbatch --array=0-599 run.sh

module load python/3.9
export SLURM_SUBMIT_DIR=/home/users/yalan/hidmed
cd $SLURM_SUBMIT_DIR

python3 run_distributed.py --job $SLURM_ARRAY_TASK_ID
