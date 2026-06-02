#!/bin/bash
# echo "开始执行 BevGrid pretraining ..."
# BevGrid
# python pretrain.py \
#     --name bevgrid-with-noise \
#     --bev_name bevgrid_ng_1b-2_with_nosie \
#     --config_file config/text_point_point/tpp_tp_bg_pp_bc.yaml \
#     > ./logs/bevgrid_pre_training_with_nosie.log 2>&1

# echo "开始执行 BevGrid-S data process ..."
# # BevGrid-S
#  # data process
# python generate_bev.py \
#     --bev_save_path /opt/data/private/dataset/nuscenes-trainval/bev/bevgrid_S-with_nosie \
#     --config_file utils/config/generate_bev_box.yaml \
#     --lower_bound -1 --train --val \
#     > ./logs/bevgrid_S_data_process_ng1_lb-1_with_nosie.log 2>&1
# # 不写bp_save_path, 分开存储point, box
# python generate_box_point_from_bev.py \
#     --config_file utils/config/generate_bev_box.yaml --train --val --filter \
#     --bev_name bevgrid_S-with_nosie \
#     > ./logs/bevgrid_S_with_nosie_box_point.log 2>&1

# echo "开始执行 BevGrid-S-1 pretraining ..."
# # S-1
# python pretrain.py \
#     --name bevgrid-s_point-1-with-noise_first_10_epoch \
#     --bev_name bevgrid_S-with_nosie \
#     --box_point_name bevgrid_S-with_nosie_ms3_f10 \
#     --config_file config/text_point_point/tpp_tp_bm_point_pp_bc.yaml \
#     --num_epochs 10 \
#     --conv 1 > ./logs/bevgrid-pretrianing-s_point-1-with-noise.log 2>&1

# python pretrain.py \
#     --name bevgrid-s_point-1-with-noise_second_10_epoch \
#     --bev_name bevgrid_S-with_nosie \
#     --box_point_name bevgrid_S-with_nosie_ms3_f10 \
#     --config_file config/text_point_point/tpp_tp_bm_box_pp_bc.yaml \
#     --resume_path /opt/data/private/output/pretrain/bevgrid-s_point-1-with-noise_first_10_epoch/290526-1826/ckpt/epoch=9-train_loss=1.25.ckpt \
#     --num_epochs 20 \
#     --conv 1 >> ./logs/bevgrid-pretrianing-s_point-1-with-noise.log 2>&1
# S-3 
# echo "开始执行 BevGrid-S-3 pretraining ..."

# python pretrain.py \
#     --name bevgrid-s_point-3-with-noise_first_10_epoch \
#     --bev_name bevgrid_S-with_nosie \
#     --box_point_name bevgrid_S-with_nosie_ms3_f10 \
#     --config_file config/text_point_point/tpp_tp_bm_point_pp_bc.yaml \
#     --num_epochs 10 \
#     --conv 3 > ./logs/bevgrid-pretrianing-s_point-3-with-noise.log 2>&1

# python pretrain.py \
#     --name bevgrid-s_point-3-with-noise_second_10_epoch \
#     --bev_name bevgrid_S-with_nosie \
#     --box_point_name bevgrid_S-with_nosie_ms3_f10 \
#     --config_file config/text_point_point/tpp_tp_bm_box_pp_bc.yaml \
#     --resume_path /opt/data/private/output/pretrain/bevgrid-s_point-3-with-noise_first_10_epoch/300526-0449/ckpt/epoch=9-train_loss=1.25.ckpt \
#     --num_epochs 20 \
#     --conv 3 >> ./logs/bevgrid-pretrianing-s_point-3-with-noise.log 2>&1



# BEVRecon
# data process
# pre-training
# BEVRecon-D
echo "BevRecon-D start "
python pretrain_mae.py \
    --name bevrecon-with-noise-D \
    --bev_name bevgrid_S-with_nosie \
    --box_point_name bevgrid_S-with_nosie_ms3_f10 \
    --config_file config_mae/distill/distill.yaml \
    > ./logs/bevrecon-pretraining-with-noise-D-with-noise.log 2>&1

echo "BevRecon-R start "

# BEVRecon-R
python pretrain_mae.py \
    --name bevrecon-with-noise-R \
    --bev_name bevgrid_S-with_nosie \
    --box_point_name bevgrid_S-with_nosie_ms3_f10 \
    --config_file config_mae/text_point_point/mae_mse_sam_mask.yaml \
    > ./logs/bevrecon-pretraining-with-noise-R-with-noise.log 2>&1

echo "BevRecon-DR start "

# BEVRecon-DR
python pretrain_mae.py \
    --name bevrecon-with-noise-DR \
    --bev_name bevgrid_S-with_nosie \
    --box_point_name bevgrid_S-with_nosie_ms3_f10 \
    --config_file config_mae/text_point_point/mae_mse_distill.yaml \
    > ./logs/bevrecon-pretraining-with-noise-DR-with-noise.log 2>&1

echo "BevRecon全部执行完成！ "