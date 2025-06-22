import cv2
import argparse
from pathlib import Path
from utils.logger import make_logger
from utils.config import generate_config, log_config
import os
import os.path as osp
import copy
import numpy as np
import torch
from torch.nn.parallel import DataParallel
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms.functional import resized_crop


from model import (
    Preprocessing,
    maskClipFeatureExtractor,
)


def get_image(scene_id):
    dataset_root = '/opt/data/private/dataset/nuscenes-mini'
    nusc = NuScenes(version="v1.0-mini", dataroot=dataset_root, verbose=False)

    scene = nusc.scene[scene_id]
    sample_token = scene['first_sample_token']
    sample = nusc.get('sample', sample_token)

    cam_front_token = sample['data']['CAM_FRONT']
    cam_front_image_path = nusc.get_sample_data_path(cam_front_token)
    image = np.array(Image.open(cam_front_image_path))
    return image

def resized_crop_adapt_maskclip(images):
    crop_size=(224, 416)
    crop_range=[0.3, 1.0]
    crop_ratio=(14.0 / 9.0, 17.0 / 9.0)
    imgs = torch.empty(
            (images.shape[0], 3) + tuple(crop_size), dtype=torch.float32
        )
    for id, img in enumerate(images):
        i, j, h, w = RandomResizedCrop.get_params(
                img, crop_range, crop_ratio
            )
        imgs[id] = resized_crop(
            img, i, j, h, w, crop_size
        )
    
    return imgs

def get_image_pred(images, model_images):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = images.to(device)
    _, images_pred = model_images(images)
    return images_pred


def parse_config():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument("--config_file", type=str, default='/opt/data/private/CSBEV/utils/config/generate_bev_box_mini.yaml', help="specify the config for processing point to bev")
    args = parser.parse_args()
    
    config = generate_config(args.config_file)

    return config

def main():
    config = parse_config()

    scene_id = 0
    class_list = [0, 1, 2, 3, 4, 5]

    image = get_image(scene_id)
    images = []
    images.append(image.astype(np.float32) / 255.0)
    images = torch.tensor(np.array(images, dtype=np.float32).transpose(0, 3, 1, 2))
    images = resized_crop_adapt_maskclip(images)
    model_images = maskClipFeatureExtractor(config, preprocessing=Preprocessing())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_images.to(device)

    images_pred = get_image_pred(images, model_images)

    image = images[0].cpu().numpy()
    if image.dtype != np.uint8:
        image = (image * 255).clip(0, 255).astype(np.uint8)
    image = np.transpose(image, (1, 2, 0))  
    image_pred = images_pred[0].cpu().numpy()

    h, w, _ = image.shape
    color_map = {
        0: (255, 105, 180, 100),   # Hotpink
        1: (255, 140, 0, 100),   # Darkorange
        2: (135, 206, 235, 100),   # Lightskyblue
        3: (255, 255, 0, 100), # Yellow
        4: (0, 255, 127, 100), # Springgreen
        5: (123, 104, 238, 100)  # Mediumslateblue
    }
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    for value, color in color_map.items():
        mask = (image_pred == value)
        overlay[mask] = color
    
    overlay_rgb = overlay[:, :, :3]
    alpha = overlay[:, :, 3] / 255.0
    output = image.copy()
    for c in range(3):
        output[:, :, c] = (1 - alpha) * image[:, :, c] + alpha * overlay_rgb[:, :, c]

    cv2.imwrite("/opt/data/private/CSBEV/picture/maskclipSeg.png", cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()