#!/bin/bash

#SBATCH --job-name=sobel_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=sobel_eval%j.out

module purge
module load anaconda3/5.3.1
module load cudnn/9.0v7.0.5
source activate dl_project

timestamp=$(date +"%Y%m%d_%H%M%S")

python -u eval_linear.py \
	  --data /scratch/drr342/dl/project/data/ssl_data_96/supervised/ \
	  --model /scratch/drr342/dl/project/models/sobel/checkpoint.pth.tar \
	  --conv 4 \
	  --exp /scratch/drr342/dl/project/results/sobel \
	  --workers 4 \
	  --epochs 50 \
	  --batch_size 256 \
	  --verbose \
	  &> DeepCluster_Eval_progress_$timestamp.txt
