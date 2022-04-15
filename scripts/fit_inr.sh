#! /bin/bash
if [ ! -d /data/vision/polina/users/clintonw/code/diffcoord/results/$1 ]
then
    mkdir /data/vision/polina/users/clintonw/code/diffcoord/results/$1
fi
sbatch <<EOT
#! /bin/bash
#SBATCH --partition=gpu
#SBATCH --time=0
#SBATCH -J $1
#SBATCH --gres=gpu:1
#SBATCH -e /data/vision/polina/users/clintonw/code/diffcoord/results/$1/err.txt
#SBATCH -o /data/vision/polina/users/clintonw/code/diffcoord/results/$1/out.txt
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --exclude=zaatar,anise,mint,clove

cd /data/vision/polina/users/clintonw/code/diffcoord/inrnet
source .bashrc
python fit_inr.py -j=$1 -c=$2 -s=$3
exit()
EOT
