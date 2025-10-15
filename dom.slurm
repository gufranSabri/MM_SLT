#!/bin/bash
#SBATCH --job-name=ft_dom
#SBATCH --output=slurm/%x.%j.out
#SBATCH --error=slurm/%x.%j.err
#SBATCH --open-mode=append
#SBATCH --account=aip-lsigal
#SBATCH --mail-user=g202302610@kfupm.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=70G
#SBATCH --time=1-00:00:00

module load python/3.11.5 cuda/12.2 gcc arrow
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

bash ./scripts/install.sh

huggingface-cli login --token token

python main_ddp.py --work-dir ./work_dir/llama2_7B_m1 --config ./configs/baseline_dom_ddp.yaml