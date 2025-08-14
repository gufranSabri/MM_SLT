# salloc --gpus-per-node=l40s:1 --cpus-per-task=6 --mem=40G --time=0-3:00 --account=aip-lsigal

module load python/3.11.5 cuda/12.2
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install torch && pip install pyyaml && pip install numpy && pip install tqdm && pip install nltk && pip install transformers==4.55.0 && pip install trl==0.20.0 && pip install peft==0.17.0 && pip install wandb==0.16.0 && pip install rouge_score && pip install sacrebleu