#! /bin/bash
#SBATCH --partition=gpu
#SBATCH --time=0
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --exclude=zaatar,anise,mint,clove
#SBATCH --array=1-4
#SBATCH -e /data/vision/polina/users/clintonw/code/inrnet/results/%J_%a.err
#SBATCH -o /data/vision/polina/users/clintonw/code/inrnet/results/%J_%a.out

cd /data/vision/polina/users/clintonw/code/inrnet/inrnet
source .bashrc
python train.py -j=$1_$SLURM_ARRAY_TASK_ID -c=$2 --sweep_id=$3
