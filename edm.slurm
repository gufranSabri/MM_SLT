#!/bin/bash
#SBATCH --job-name=ft_t5
#SBATCH --output=slurm/%x.%j.out
#SBATCH --error=slurm/%x.%j.err
#SBATCH --open-mode=append
#SBATCH --account=aip-lsigal
#SBATCH --mail-user=g202302610@kfupm.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=6
#SBATCH --mem=50G
#SBATCH --time=1-00:00:00

module load python/3.11.5 cuda/12.2 gcc arrow
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

bash ./scripts/install.sh

# Timestamp for unique work directory
timestamp=$(date +"%d_%H_%M_%S")
work_dir="./work_dir/t5xl_m1_${timestamp}"

export PYTHONDONTWRITEBYTECODE=1
python main_ddp.py --work-dir "$work_dir" --config ./configs/baseline_edm_ddp.yaml
python test.py --work-dir "$work_dir" --config "$work_dir/baseline_edm_ddp.yaml"
