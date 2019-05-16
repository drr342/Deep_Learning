#!/bin/bash

#SBATCH --job-name=sobel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=sobel_%j.out

module purge
module load anaconda3/5.3.1
module load cudnn/9.0v7.0.5
source activate dl_project

timestamp=$(date +"%Y%m%d_%H%M%S")

python -u main.py \
	  --arch alexnet \
	  --sobel \
	  --clustering Kmeans \
	  --nmb_cluster 10000 \
	  --workers 2 \
	  --start_epoch 13 \
	  --epochs 50 \
	  --batch 256 \
	  --verbose \
	  --exp /scratch/drr342/dl/project/models/sobel \
	  --resume /scratch/drr342/dl/project/models/sobel/checkpoint.pth.tar \
	  /scratch/drr342/dl/project/data/ssl_data_96/unsupervised/ \
	  &> DeepCluster_progress_$timestamp.txt

# --resume /scratch/drr342/dl/project/models/checkpoint.pth.tar \
