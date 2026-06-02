#!/bin/bash
# For calibration noise experiment
# cd downstream
# finetun
    # --cfg_file', type=str, default='/opt/data/private/code/BEVConMask/downstream/config/kitti/second.yaml
    # BEVGrid
    python train.py \
    --name bevgrid-with-noise-1st \
    --pretrained_model /opt/data/private/output/pretrain/bevgrid-with-noise/280526-1511/model_epoch20.pt \
    --data_skip_step 20 > ../logs/bevgrid-with-noise-finetuning-1st.log 2>&1
    python train.py \
    --name bevgrid-with-noise-2st \
    --pretrained_model /opt/data/private/output/pretrain/bevgrid-with-noise/280526-1511/model_epoch20.pt \
    --data_skip_step 20 > ../logs/bevgrid-with-noise-finetuning-2st.log 2>&1
    python train.py \
    --name bevgrid-with-noise-3st \
    --pretrained_model /opt/data/private/output/pretrain/bevgrid-with-noise/280526-1511/model_epoch20.pt \
    --data_skip_step 20 > ../logs/bevgrid-with-noise-finetuning-3st.log 2>&1
    
    # BEVGrid-S-1
    python train.py\
    --name bevgrid-s_point-1-with-noise-percent-5-1st \
    --pretrained_model /opt/data/private/output/pretrain/bevgrid-s_point-1-with-noise_second_10_epoch/300526-1808/model_epoch20.pt \
    --data_skip_step 20 \
    > ../logs/bevgrid-s-1-with-noise-finetuning-1st.log 2>&1
    python train.py\
    --name bevgrid-s_point-1-with-noise-percent-5-2st \
    --pretrained_model /opt/data/private/output/pretrain/bevgrid-s_point-1-with-noise_second_10_epoch/300526-1808/model_epoch20.pt \
    --data_skip_step 20 \
    > ../logs/bevgrid-s-1-with-noise-finetuning-2st.log 2>&1
    python train.py\
    --name bevgrid-s_point-1-with-noise-percent-5-3st \
    --pretrained_model /opt/data/private/output/pretrain/bevgrid-s_point-1-with-noise_second_10_epoch/300526-1808/model_epoch20.pt \
    --data_skip_step 20 \
    > ../logs/bevgrid-s-1-with-noise-finetuning-3st.log 2>&1

    # BEVGrid-S-3
    python train.py \
    --name bevgrid-s_point-3-with-noise-percent-5-1st \
    --pretrained_model /opt/data/private/output/pretrain/bevgrid-s_point-3-with-noise_second_10_epoch/310526-0437/model_epoch20.pt \
    --data_skip_step 20 \
    > ../logs/bevgrid-s-3-with-noise-finetuning-1st.log 2>&1
    python train.py \
    --name bevgrid-s_point-3-with-noise-percent-5-2st \
    --pretrained_model /opt/data/private/output/pretrain/bevgrid-s_point-3-with-noise_second_10_epoch/310526-0437/model_epoch20.pt \
    --data_skip_step 20 \
    > ../logs/bevgrid-s-3-with-noise-finetuning-2st.log 2>&1
    python train.py \
    --name bevgrid-s_point-3-with-noise-percent-5-3st \
    --pretrained_model /opt/data/private/output/pretrain/bevgrid-s_point-3-with-noise_second_10_epoch/310526-0437/model_epoch20.pt \
    --data_skip_step 20 \
    > ../logs/bevgrid-s-3-with-noise-finetuning-3st.log 2>&1

    # BEVRecon-D
    python train.py \
    --name bevrecon-with-noise-D-1st \
    --pretrained_model /opt/data/private/output/mae/pretrain/bevrecon-with-noise-D/310526-1628/model_epoch20.pt \
    --data_skip_step 20 \
    > ../logs/bevrecon-with-noise-D-finetuning-1st.log 2>&1
    python train.py \
    --name bevrecon-with-noise-D-2st \
    --pretrained_model /opt/data/private/output/mae/pretrain/bevrecon-with-noise-D/310526-1628/model_epoch20.pt \
    --data_skip_step 20 \
    > ../logs/bevrecon-with-noise-D-finetuning-2st.log 2>&1
    python train.py \
    --name bevrecon-with-noise-D-3st \
    --pretrained_model /opt/data/private/output/mae/pretrain/bevrecon-with-noise-D/310526-1628/model_epoch20.pt \
    --data_skip_step 20 \
    > ../logs/bevrecon-with-noise-D-finetuning-3st.log 2>&1
    # BEVRecon-R
    python train.py \
    --name bevrecon-with-noise-R-1st \
    --pretrained_model /opt/data/private/output/mae/pretrain/bevrecon-with-noise-R/010626-0650/model_epoch20.pt \
    --data_skip_step 20 \
    > ../logs/bevrecon-with-noise-R-finetuning-1st.log 2>&1
    python train.py \
    --name bevrecon-with-noise-R-2st \
    --pretrained_model /opt/data/private/output/mae/pretrain/bevrecon-with-noise-R/010626-0650/model_epoch20.pt \
    --data_skip_step 20 \
    > ../logs/bevrecon-with-noise-R-finetuning-2st.log 2>&1
    python train.py \
    --name bevrecon-with-noise-R-3st \
    --pretrained_model /opt/data/private/output/mae/pretrain/bevrecon-with-noise-R/010626-0650/model_epoch20.pt \
    --data_skip_step 20 \
    > ../logs/bevrecon-with-noise-R-finetuning-3st.log 2>&1
    # BEVRecon-DR
    python train.py \
    --name bevrecon-with-noise-DR-1st \
    --pretrained_model /opt/data/private/output/mae/pretrain/bevrecon-with-noise-DR/010626-2250/model_epoch20.pt \
    --data_skip_step 20 \
    > ../logs/bevrecon-with-noise-DR-finetuning-1st.log 2>&1
    python train.py \
    --name bevrecon-with-noise-DR-2st \
    --pretrained_model /opt/data/private/output/mae/pretrain/bevrecon-with-noise-DR/010626-2250/model_epoch20.pt \
    --data_skip_step 20 \
    > ../logs/bevrecon-with-noise-DR-finetuning-2st.log 2>&1
    python train.py \
    --name bevrecon-with-noise-DR-3st \
    --pretrained_model /opt/data/private/output/mae/pretrain/bevrecon-with-noise-DR/010626-2250/model_epoch20.pt \
    --data_skip_step 20 \
    > ../logs/bevrecon-with-noise-DR-finetuning-3st.log 2>&1
# test after fune-tune
    # '--cfg_file', type=str, default='/opt/data/private/code/BEVConMask/downstream/config/kitti/second.yaml'
    # BEVGrid
    python test.py \
    --name bevgrid-with-noise-1st  \
    --ckpt /opt/data/private/output/finetune/bevgrid-with-noise-1st/ \
    > ../logs/bevgrid-with-noise-test-1st.log 2>&1
    python test.py \
    --name bevgrid-with-noise-2st  \
    --ckpt /opt/data/private/output/finetune/bevgrid-with-noise-2st/ \
    > ../logs/bevgrid-with-noise-2st.log 2>&1
    python test.py \
    --name bevgrid-with-noise-3st  \
    --ckpt /opt/data/private/output/finetune/bevgrid-with-noise-3st/ \
    > ../logs/bevgrid-with-noise-test-3st.log 2>&1

    # BEVGrid-S-1
    python test.py \
    --name bevgrid-s-1-with-noise-percent-5-1st \
    --ckpt /opt/data/private/output/finetune/bevgrid-s_point-1-with-noise-percent-5-1st/ \
    > ../logs/bevgrid-s-1-with-noise-test-1st.log 2>&1
    python test.py \
    --name bevgrid-s-1-with-noise-percent-5-2st \
    --ckpt /opt/data/private/output/finetune/bevgrid-s_point-1-with-noise-percent-5-2st/ \
    > ../logs/bevgrid-s-1-with-noise-test-2st.log 2>&1
    python test.py \
    --name bevgrid-s-1-with-noise-percent-5-3st \
    --ckpt /opt/data/private/output/finetune/bevgrid-s_point-1-with-noise-percent-5-3st/ \
    > ../logs/bevgrid-s-1-with-noise-test-3st.log 2>&1
    # BEVGrid-S-3
    python test.py \
    --name bevgrid-s-3-with-noise-percent-5-1st \
    --ckpt /opt/data/private/output/finetune/bevgrid-s_point-3-with-noise-percent-5-1st \
    > ../logs/bevgrid-s-3-with-noise-test-1st.log 2>&1
    python test.py \
    --name bevgrid-s-3-with-noise-percent-5-2st \
    --ckpt /opt/data/private/output/finetune/bevgrid-s_point-3-with-noise-percent-5-2st \
    > ../logs/bevgrid-s-3-with-noise-test-2st.log 2>&1
    python test.py \
    --name bevgrid-s-3-with-noise-percent-5-2st \
    --ckpt /opt/data/private/output/finetune/bevgrid-s_point-3-with-noise-percent-5-2st \
    > ../logs/bevgrid-s-3-with-noise-test-2st.log 2>&1

    # BEVRecon-D
    python test.py \
    --name bevrecon-with-noise-D-1st \
    --ckpt /opt/data/private/output/finetune/bevrecon-with-noise-D-1st \
    > ../logs/bevrecon-with-noise-D-1st.log 2>&1
    python test.py \
    --name bevrecon-with-noise-D-2st \
    --ckpt /opt/data/private/output/finetune/bevrecon-with-noise-D-2st \
    > ../logs/bevrecon-with-noise-D-2st.log 2>&1
    python test.py \
    --name bevrecon-with-noise-D-3st \
    --ckpt /opt/data/private/output/finetune/bevrecon-with-noise-D-3st \
    > ../logs/bevrecon-with-noise-D-3st.log 2>&1
    # BEVRecon-R
    python test.py \
    --name bevrecon-with-noise-R-1st \
    --ckpt /opt/data/private/output/finetune/bevrecon-with-noise-R-1st \
    > ../logs/bevrecon-with-noise-R-1st.log 2>&1
    python test.py \
    --name bevrecon-with-noise-R-2st \
    --ckpt /opt/data/private/output/finetune/bevrecon-with-noise-R-2st \
    > ../logs/bevrecon-with-noise-R-2st.log 2>&1
    python test.py \
    --name bevrecon-with-noise-R-3st \
    --ckpt /opt/data/private/output/finetune/bevrecon-with-noise-R-3st \
    > ../logs/bevrecon-with-noise-R-3st.log 2>&1
    # BEVRecon-DR
    python test.py \
    --name bevrecon-with-noise-DR-1st \
    --ckpt /opt/data/private/output/finetune/bevrecon-with-noise-DR-1st \
    > ../logs/bevrecon-with-noise-DR-1st.log 2>&1
    python test.py \
    --name bevrecon-with-noise-DR-2st \
    --ckpt /opt/data/private/output/finetune/bevrecon-with-noise-DR-2st \
    > ../logs/bevrecon-with-noise-DR-2st.log 2>&1
    python test.py \
    --name bevrecon-with-noise-DR-3st \
    --ckpt /opt/data/private/output/finetune/bevrecon-with-noise-DR-3st \
    > ../logs/bevrecon-with-noise-DR-3st.log 2>&1
