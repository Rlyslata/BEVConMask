# BEVConMask: A Text-Driven Self-Supervised BEV Pretraining Framework via Contrastive Learning and Masked Modeling

# Installation

```shell
conda create -n bevconmask python=3.8.3

pip install -r requirements.txt

# Note that we should manually add the following function to the class "LidarPointCloud"
# in "miniconda3/envs/{your environment name}/lib/python{your python version}/site-packages/nuscenes/utils/data_classes.py"
class LidarPointCloud(PointCloud):
    @classmethod
    def from_points(cls, points) -> 'LidarPointCloud':
        return cls(points.T)
```

# Data Preparation

In this paper, we conduct experiments on [Nuscenes](https://www.nuscenes.org/nuscenes#overview) and [KITTI3D](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).

**Step 1.** Download the NuScenes and KITTI3D datasets and place them in the dataset folder. For dataset preparation, please refer to [GETTING_STARTED](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md) or follow the steps below.
```shell
# pretrain: nuscenes-mini and nuscenes-trainval

# finetune: kitti3D
cd downstream
python datasets/kitti_dataset.py --cfg_file config/kitti/kitti_dataset.yaml --data_path your_kitti3D_path
```

**Step 2.** Download and convert the CLIP models.
```shell
# obtain ViT16_clip_backbone.pth and ViT16_clip_weights.pth
python utils/convert_clip_weights.py --model ViT16 --backbone
python utils/convert_clip_weights.py --model ViT16
```

**Step 3.** Prepare the CLIP's text embeddings of the NuScenes dataset.
```shell
# obtain nuscenes_ViT16_clip_text.pth
python utils/prompt_engineering.py --model ViT16 --class-set nuscenes
```

**Step 4.** Download the EfficientSAM-Ti model from [EfficientSAM](https://github.com/yformer/EfficientSAM?tab=readme-ov-file).

# Pre-training on NuScenes

## BEVGrid

**Step 1.** Data preprocessing.
```shell
# Generate a bev_map with a height range of -2 to 1 and a grid size of 1.
# ng1_lb-2.0
python generate_bev.py --bev_save_path your_path --config_file utils/config/generate_bev_box.yaml --lower_bound -2 --train --val
```

**Step 2.** Pre-training.
```shell
# Use the bev_map above for pre-training.
python pretrain.py --name bevgrid --config_file config/text_point_point/tpp_tp_bg_pp_bc.yaml
```

## BEVGrid-S

**Step 1.** Data preprocessing.
```shell
# Generate a bev_map with a height range of -1 to 1 and a grid size of 1.
# ng1_lb-1
python generate_bev.py --bev_save_path your_path --config_file utils/config/generate_bev_box.yaml --lower_bound -1 --train --val

# Generate box and point prompts from the bev_map above.
# ng1_lb-1_ms3_f10
python generate_box_point_from_bev.py --bp_save_path your_path --config_file utils/config/generate_bev_box.yaml --train --val --filter
```

**Step 2.** Pre-training.
```shell
# Use point prompts for pre-training
python pretrain.py --name bevgrid-s_point --config_file config/text_point_point/tpp_tp_bm_point_pp_bc.yaml --conv 1 or 3

# Use box prompts for pre-training
python pretrain.py --name bevgrid-s_box --config_file config/text_point_point/tpp_tp_bm_box_pp_bc.yaml --conv 1 or 3

# The above two are single prompt strategies. If you want to use prompt switch strategies, first train with one prompt for 10 epochs, then read the checkpoint and train with the other prompt.
```

## BEVRecon

**Step 1.** Data preprocessing. Please prepare the teacher model BEVGrid-S-3.

**Step 2.** Pre-training.
```shell
python pretrain_mae.py --name bevrecon --config_file config_mae/text_point_point/mae_mse_distill.yaml
```

# Fine-tuning on KITTI3D

**Step 1.** Fine-tuning.
```shell
cd downstream

# second
# 100%
python train.py --name your_name --pretrained_model your_model
# 50%
python train.py --name your_name --pretrained_model your_model --data_skip_step 2
# 20%
python train.py --name your_name --pretrained_model your_model --data_skip_step 5
# 10%
python train.py --name your_name --pretrained_model your_model --data_skip_step 10
# 5%
python train.py --name your_name --pretrained_model your_model --data_skip_step 20

# pv-rcnn++
# 100%
python train.py --name your_name --pretrained_model your_model --cfg_file config/kitti/pv_rcnn_plusplus.yaml
# 50%
python train.py --name your_name --pretrained_model your_model --data_skip_step 2 --cfg_file config/kitti/pv_rcnn_plusplus.yaml
# 20%
python train.py --name your_name --pretrained_model your_model --data_skip_step 5 --cfg_file config/kitti/pv_rcnn_plusplus.yaml
# 10%
python train.py --name your_name --pretrained_model your_model --data_skip_step 10 --cfg_file config/kitti/pv_rcnn_plusplus.yaml
# 5%
python train.py --name your_name --pretrained_model your_model --data_skip_step 20 --cfg_file config/kitti/pv_rcnn_plusplus.yaml
```

**Step 2.** Test.
```shell
# second
python test.py --name your_name --pretrained_model your_model --ckpt your_ckpt

# pv-rcnn++
python test.py --name your_name --pretrained_model your_model --ckpt your_ckpt --cfg_file config/kitti/pv_rcnn_plusplus.yaml
```

# Acknowledgement

Part of the codebase has been adapted from [BEVContrast](https://github.com/valeoai/BEVContrast), [CLIP2Scene](https://github.com/runnanchen/CLIP2Scene), [EfficientSAM](https://github.com/yformer/EfficientSAM?tab=readme-ov-file), [BEV-MAE](https://github.com/VDIGPKU/BEV-MAE) and [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).