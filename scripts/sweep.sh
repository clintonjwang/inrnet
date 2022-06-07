#! /bin/bash
#SBATCH --partition=gpu
#SBATCH --time=0
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --exclude=zaatar,anise,mint,clove
#SBATCH -e /data/vision/polina/users/clintonw/code/inrnet/temp/logs/%A_%a.err
#SBATCH -o /data/vision/polina/users/clintonw/code/inrnet/temp/logs/%A_%a.out

cd /data/vision/polina/users/clintonw/code/inrnet/inrnet
python train.py --job_id=${SLURM_JOB_NAME}_${SLURM_ARRAY_TASK_ID} -c=$conf --sweep_id=$SLURM_JOB_NAME
