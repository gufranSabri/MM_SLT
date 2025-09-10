python /home/ahmedubc/projects/aip-lsigal/ahmedubc/MM_SLT/preprocess/SP_FT/vivit_extract_feature.py \
    --anno_root /home/ahmedubc/projects/aip-lsigal/ahmedubc/MM_SLT/preprocess/phoenix2014-T \
    --model_name google/vivit-b-16x2-kinetics400 \
    --video_root /home/ahmedubc/projects/aip-lsigal/ahmedubc/datasets/phoenix2014T/ \
    --cache_dir /home/ahmedubc/projects/aip-lsigal/ahmedubc/datasets/vivit_sp/sp \
    --save_dir /home/ahmedubc/projects/aip-lsigal/ahmedubc/datasets/vivit_sp/sp \
    --s2_mode s2wrapping \
    --scales 1 2 \
    --batch_size 8 \
    --device cuda:0