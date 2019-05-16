#!/bin/bash

if [ -z $1 ]; then
  echo "usage: $0 <python_script>"
  exit 1
fi

if ! [ -f $1 ]; then
  echo "File $1 not found!"
  exit 1
fi

script=$1

sbatch << EOF
#!/bin/bash

#SBATCH --job-name=dl_$script
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=dl_$script_%j.out

module purge
module load python3/intel/3.6.3
module load cudnn/9.0v7.0.5
# module load ninja/intel/1.8.2
# module load gcc/6.3.0
source ~/pyenv/py3.6.3/bin/activate

python ./$script

EOF