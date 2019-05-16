#!/bin/bash

#SBATCH --job-name=array_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=array_eval__%A_%a.out

module purge
module load anaconda3/5.3.1
module load cudnn/9.0v7.0.5
source activate dl_project

python -u eval_linear.py \
	  --data /scratch/drr342/dl/project/data/ssl_data_96/supervised/ \
	  --model /scratch/drr342/dl/project/models/checkpoint.pth.tar \
	  --conv 4 \
	  --exp /scratch/drr342/dl/project/results/ \
	  --workers 4 \
	  --epochs 30 \
	  --batch_size 256 \
	  --labeled ${SLURM_ARRAY_TASK_ID} \
	  --verbose \
	  &> DeepCluster_array_progress_${SLURM_ARRAY_TASK_ID}.txt
