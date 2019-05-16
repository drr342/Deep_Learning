#!/bin/bash

#SBATCH --job-name=dl_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --output=dl_eval_%j.out

module purge
module load python3/intel/3.6.3
module load cudnn/9.0v7.0.5
source ~/pyenv/py3.6.3/bin/activate

python eval.py --data_dir /scratch/drr342/dl/project/data/ssl_data_96