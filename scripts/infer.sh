#! /bin/bash
if [ ! -d /data/vision/polina/users/clintonw/code/inrnet/results/$1 ]
then
    mkdir /data/vision/polina/users/clintonw/code/inrnet/results/$1
fi
sbatch <<EOT
#! /bin/bash
#SBATCH --partition=${4:-gpu}
#SBATCH --time=0
#SBATCH -J $1
#SBATCH --gres=gpu:1
#SBATCH -e /data/vision/polina/users/clintonw/code/inrnet/results/$1/err.txt
#SBATCH -o /data/vision/polina/users/clintonw/code/inrnet/results/$1/out.txt
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --exclude=zaatar,anise,mint,clove

cd /data/vision/polina/users/clintonw/code/inrnet/inrnet
source .bashrc
python infer.py -j=$1 -c=$2 --target_job=$3
exit()
EOT

#
#bergamot,perilla,caraway,cassia
# --mem-per-gpu=48G
# #SBATCH --exclusive
# ,zaatar,peppermint