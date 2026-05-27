#!/bin/bash
echo "开始执行 BEVRecon-R ..."
# 1. 运行 BEVRecon-R
python pretrain_mae.py \
    --name bevrecon-with-noise-R \
    --bev_name bevgrid_S-with_nosie \
    --box_point_name bevgrid_S-with_nosie_ms3_f10 \
    --config_file config_mae/text_point_point/mae_mse_sam_mask.yaml \
    > ./logs/bevrecon-pretraining-with-noise-R-with-noise.log 2>&1

echo "BEVRecon-R 完成，开始执行 BEVRecon-DR ..."

# 2. 运行 BEVRecon-DR
python pretrain_mae.py \
    --name bevrecon-with-noise-DR \
    --bev_name bevgrid_S-with_nosie \
    --box_point_name bevgrid_S-with_nosie_ms3_f10 \
    --config_file config_mae/text_point_point/mae_mse_sam_point_switch.yaml \
    > ./logs/bevrecon-pretraining-with-noise-DR-with-noise.log 2>&1

echo "全部执行完成！"