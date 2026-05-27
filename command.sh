# 标定噪声实验
    # BEVGrid 完成
        # data process
            nohup python generate_bev.py --bev_save_path /opt/data/private/dataset/nuscenes-trainval/bev/bevgrid_ng_1b-2_with_nosie --config_file utils/config/generate_bev_box.yaml --lower_bound -2 --train --val > ./logs/bevgrid_data_process_bevgrid_ng_1b-2_with_nosie.log 2>&1 &

        # pre-training
            # SAVE_PATH: "/opt/data/private/output/pretrain"
            # PyTorch Lightning 的 default_root_dir 被设置为同一个基础目录（{MODEL.SAVE_PATH}/{name}/{时间戳}）
            # "/opt/data/private/output/pretrain/bevgrid-with-noise/{时间戳}/ckpt/*.ckpt"
            nohup python pretrain.py --name bevgrid-with-noise --bev_name bevgrid_ng_1b-2_with_nosie --config_file config/text_point_point/tpp_tp_bg_pp_bc.yaml > ./logs/bevgrid_pre_training_with_nosie.log 2>&1 &
        
        # fine-tuning
            cd downstream
            # 5%
                nohup python train.py --name bevgrid-with-noise --pretrained_model /opt/data/private/output/pretrain/bevgrid-with-noise/170526-2359/model_epoch20.pt --data_skip_step 20 > ../logs/bevgrid-with-noise-finetuning.log 2>&1 &
        # test(SECOND) 
            nohup python test.py --name bevgrid-with-noise  --ckpt /opt/data/private/output/finetune/bevgrid-with-noise/220526-1249/ckpt/checkpoint_epoch_20.pth > ../logs/bevgrid-with-noise-test.log 2>&1 &


    # BEVGrid-S 
        # data process
            nohup python generate_bev.py --bev_save_path /opt/data/private/dataset/nuscenes-trainval/bev/bevgrid_S-with_nosie --config_file utils/config/generate_bev_box.yaml --lower_bound -1 --train --val > ./logs/bevgrid_S_data_process_ng1_lb-1_with_nosie.log 2>&1 &
            # 不写bp_save_path, 分开存储point, box
            nohup python generate_box_point_from_bev.py \
            --config_file utils/config/generate_bev_box.yaml --train --val --filter \
            --bev_name bevgrid_S-with_nosie \
            > ./logs/bevgrid_S_with_nosie_box_point.log 2>&1 &
        
        # pre-training
            # bevgrid-s_point-1-with-noise
                # S-1 配置文件弄错了,没有选择switch，有时间再搞
                nohup python pretrain.py --name bevgrid-s_point-1-with-noise --bev_name bevgrid_S-with_nosie --box_point_name bevgrid_S-with_nosie_ms3_f10 --config_file config/text_point_point/tpp_tp_bm_point_pp_bc.yaml --conv 1 > ./logs/bevgrid-pretrianing-s_point-1-with-noise.log 2>&1 &
                
                nohup python pretrain.py --name bevgrid-s_point-3-with-noise --bev_name bevgrid_S-with_nosie --box_point_name bevgrid_S-with_nosie_ms3_f10 --config_file config/text_point_point/tpp_tp_bm_box_point_and_switch_pp_bc.yaml --conv 3 > ./logs/bevgrid-pretrianing-s_point-3-with-noise.log 2>&1 &
            
        # fine tuning
            cd downstream
            # 5%
                nohup python train.py --name bevgrid-s_point-1-with-noise-percent-5 --pretrained_model /opt/data/private/output/pretrain/bevgrid-s_point-1-with-noise/220526-1819/model_epoch20.pt --data_skip_step 20 > ../logs/bevgrid-s-1-with-noise-finetuning.log 2>&1 &
                nohup python train.py --name bevgrid-s_point-3-with-noise-percent-5 --pretrained_model /opt/data/private/output/pretrain/bevgrid-s_point-3-with-noise/230526-1505/model_epoch20.pt --data_skip_step 20 > ../logs/bevgrid-s-3-with-noise-finetuning.log 2>&1 &
        # test(SECOND)
                nohup python test.py --name bevgrid-s-1-with-noise-percent-5 --ckpt /opt/data/private/output/finetune/bevgrid-s_point-1-with-noise-percent-5/240526-1419/ckpt/checkpoint_epoch_20.pth > ../logs/bevgrid-s-1-with-noise-test.log 2>&1 &
                nohup python test.py --name bevgrid-s-3-with-noise-percent-5 --ckpt /opt/data/private/output/finetune/bevgrid-s_point-3-with-noise-percent-5/240526-1437/ckpt/checkpoint_epoch_20.pth > ../logs/bevgrid-s-3-with-noise-test.log 2>&1 &

    # BEVRecon
        # data process
        # pre-training
            # BEVRecon-D
            nohup python pretrain_mae.py --name bevrecon-with-noise-D --bev_name bevgrid_S-with_nosie --box_point_name bevgrid_S-with_nosie_ms3_f10 --config_file config_mae/text_point_point/mae_mse_distill.yaml > ./logs/bevrecon-pretraining-with-noise-D-with-noise.log 2>&1 &
            # BEVRecon-R
            nohup python pretrain_mae.py --name bevrecon-with-noise-R --bev_name bevgrid_S-with_nosie --box_point_name bevgrid_S-with_nosie_ms3_f10 --config_file config_mae/text_point_point/mae_mse_sam_mask.yaml > ./logs/bevrecon-pretraining-with-noise-R-with-noise.log 2>&1 &
            # BEVRecon-DR
            nohup python pretrain_mae.py --name bevrecon-with-noise-DR --bev_name bevgrid_S-with_nosie --box_point_name bevgrid_S-with_nosie_ms3_f10 --config_file config_mae/text_point_point/mae_mse_sam_point_switch.yaml > ./logs/bevrecon-pretraining-with-noise-DR-with-noise.log 2>&1 &
        # # fine-tuning
        #     # 5%
        #         # BEVRecon-D
        #         nohup python train.py --name bevrecon-with-noise-D --pretrained_model /opt/data/private/output/ --data_skip_step 20 > ../logs/bevrecon-with-noise-D-finetuning.log 2>&1 &
        #         # BEVRecon-R
        #         nohup python train.py --name bevrecon-with-noise-R --pretrained_model /opt/data/private/output/ --data_skip_step 20 > ../logs/bevrecon-with-noise-R-finetuning.log 2>&1 &
        #         # BEVRecon-DR
        #         nohup python train.py --name bevrecon-with-noise-DR --pretrained_model /opt/data/private/output/ --data_skip_step 20 > ../logs/bevrecon-with-noise-DR-finetuning.log 2>&1 &

                
        # # test 
        #     python test.py --name bevrecon-with-noise-D --ckpt /opt/data/private/output/
        #     python test.py --name bevrecon-with-noise-R --ckpt your_/opt/data/private/output/ckpt
        #     python test.py --name bevrecon-with-noise-DR --ckpt /opt/data/private/output/

# # 跨数据集实验
#     # 数据集情况
#         正在从 https://opendatalab.com/OpenDataLab/Waymo 下载
#     # waymo 数据脚本
#         准备downstream/datasets/waymo_dataset.py, 尚未完成
#     # fine-tuing
#         # No pretrain
#             是指直接用 SECOND 检测头 吗，不微调直接执行 : python test.py --name your_name --pretrained_model your_model --ckpt your_ckpt
#             那 your_model, your_ckpt 如何设置
#         # BEVContrast
#             # 对OPENPCDet不了解，不知道怎么微调，麻烦师姐指导：
#                 # 1%
#                     python train.py --name your_name --pretrained_model your_model --data_skip_step 100

#                 # 10
#                     python train.py --name your_name --pretrained_model your_model --data_skip_step 10

#                 your_model 指定为 "预训练模型.zip" 的 BEVContrast.pt, 如 checkpoints/pretrained_model/BEVContrast.pt 或 pretrain.py的到的模型checkpoint,
#                 另外，downstream/train.py 的--cfg_file默认为.../kitti/second.yml, 这里需要指定为waymo对应yml?
#                 parser.add_argument('--cfg_file', type=str, default='/opt/data/private/BEVConMask/downstream/config/kitti/second.yaml', help='specify the config for training')
#         # BEV-MAE
#             --cfg_file 使用second_res.yaml

#         # BEVGrid
            
#         # BEVGrid-S-1
#             "预训练模型.zip"没有checkpoint，需要训练获得

#         # BEVGrid-S-3

#         # BEVRecon-D
#             "预训练模型.zip"没有checkpoint，需要训练获得

#         # BEVRecon-R
#             "预训练模型.zip"没有checkpoint，需要训练获得

#         # BEVRecon-DR
#     # test
#         # 分别测试 1%, 10% 微调模型，your_model，your_model 如何设置？your_model = 预训练得到的模型，your_ckpt = 微调后的模型？
#         python test.py --name your_name --pretrained_model your_model --ckpt your_ckpt

# # 计算开销
#     OPENPCDet 或 python-lightening会输出吗？