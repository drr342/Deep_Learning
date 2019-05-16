#!/bin/bash

#SBATCH --job-name=dl_hw3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --gres=gpu:k80:1
#SBATCH --time=01:00:00
#SBATCH --output=dl_hw3__%A_%a.out

module purge
module load python3/intel/3.6.3
module load cudnn/9.0v7.0.5
source ~/pyenv/py3.6.3/bin/activate

i=1
for emsize in 100 200 400
do
	for nlayers in 1 2 4
	do
		for bptt in 16 32 64
		do
			map_emsize[$i]=$emsize
			map_nlayers[$i]=$nlayers
			map_bptt[$i]=$bptt
			i=$(( i + 1 ))
		done
	done
done

if [ $SLURM_ARRAY_TASK_ID -le 27 ]
then
	index=$SLURM_ARRAY_TASK_ID
	model=LSTM
else
	index=$(( $SLURM_ARRAY_TASK_ID - 27 ))
	model=GRU
fi

em=${map_emsize[$index]}
nl=${map_nlayers[$index]}
bp=${map_bptt[$index]}

path=/home/drr342/dl/assignments/hw3/examples/word_language_model
data_path=$path/data/wikitext-2
save_path=/home/drr342/dl/assignments/hw3/models

echo "python $path/main.py --data $data_path --save $save_path/$model'_'$em'_'$nl'_'$bp.pt --cuda --model $model --epochs 10 --emsize $em --nlayers $nl --bptt $bp"
echo ""
python $path/main.py --data $data_path --save $save_path/$model'_'$em'_'$nl'_'$bp.pt --cuda --model $model --epochs 10 --emsize $em --nlayers $nl --bptt $bp