python /home/ahmedubc/projects/aip-lsigal/ahmedubc/MM_SLT/preprocess/SP_FT/dino_extract_feature.py \
    --anno_root /home/ahmedubc/projects/aip-lsigal/ahmedubc/MM_SLT/preprocess/phoenix2014-T \
    --model_name facebook/dinov2-giant-imagenet1k-1-layer \
    --video_root /home/ahmedubc/projects/aip-lsigal/ahmedubc/datasets/phoenix2014T/ \
    --cache_dir /home/ahmedubc/projects/aip-lsigal/ahmedubc/datasets/dino_sp/sp \
    --save_dir /home/ahmedubc/projects/aip-lsigal/ahmedubc/datasets/dino_sp/sp \
    --s2_mode s2wrapping \
    --scales 1 2 \
    --batch_size 8 \
    --device cuda:0