# salloc --gpus-per-node=l40s:1 --cpus-per-task=6 --mem=60G --time=1-00:00:00 --account=aip-lsigal
# salloc --gpus-per-node=l40s:4 --cpus-per-task=6 --mem=60G --time=12:00:00 --account=aip-lsigal


module load python/3.11.5 cuda/12.2 gcc arrow && virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip && bash ./scripts/install.sh

# ========================

python main_ddp.py --work-dir ./work_dir/test --config ./configs/baseline_edm_ddp.yaml
