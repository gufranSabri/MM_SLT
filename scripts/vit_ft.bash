python /home/ahmedubc/projects/aip-lsigal/ahmedubc/MM_SLT/preprocess/SP_FT/vit_extract_feature.py \
    --anno_root /home/ahmedubc/projects/aip-lsigal/ahmedubc/MM_SLT/preprocess/phoenix2014-T \
    --model_name openai/clip-vit-large-patch14 \
    --video_root /home/ahmedubc/projects/aip-lsigal/ahmedubc/datasets/phoenix2014T/ \
    --cache_dir /home/ahmedubc/projects/aip-lsigal/ahmedubc/datasets/dump \
    --save_dir /home/ahmedubc/projects/aip-lsigal/ahmedubc/datasets/dump \
    --s2_mode s2wrapping \
    --scales 1 2 \
    --batch_size 32 \
    --device cuda:0