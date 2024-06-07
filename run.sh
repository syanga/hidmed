#!/bin/bash
############################## Submit Job ######################################
#SBATCH --time=48:00:00
#SBATCH --job-name="hidmed"
#SBATCH --output=hidmed_%j.txt
#SBATCH --error=hidmed_err_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=8G
#SBATCH --partition=normal
#####################################

# run using sbatch --array=0-5 run.sh

module load python/3.9
export SLURM_SUBMIT_DIR=/home/users/yalan/hidmed
cd $SLURM_SUBMIT_DIR

python3 run.py --job $SLURM_ARRAY_TASK_ID
