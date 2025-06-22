from generate_bev import NuscenesDataset
from generate_bev import CUSTOM_SPLIT
from generate_bev import make_dataloader
import argparse
from pathlib import Path
from utils.logger import make_logger
from utils.config import generate_config, log_config
import os
import os.path as osp
import copy
import numpy as np
import hdbscan
import pickle
import random
import torch
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

class NuscenesBEVDataset(NuscenesDataset):
    def __init__(
        self,
        phase,
        config,
        **kwargs,
    ):
        self.phase = phase
        self.dataset_root = config.DATASET.DATASET_ROOT

        self.bev_path = Path(config.DATASET.BEV_SAVE_PATH, 'bev', config.BEV_NAME)

        if "cached_nuscenes" in kwargs:
            self.nusc = kwargs["cached_nuscenes"]
        elif config.DATASET.IS_MINI:
            self.nusc = NuScenes(
                version="v1.0-mini", dataroot=self.dataset_root, verbose=False
            )
        else:
            self.nusc = NuScenes(
                version="v1.0-trainval", dataroot=self.dataset_root, verbose=False
            )

        self.frame_list = list()

        if phase in ("train", "val", "test"):
            phase_scenes = create_splits_scenes()[phase]
        elif phase == "parametrizing":
            phase_scenes = list(
                set(create_splits_scenes()["train"]) - set(CUSTOM_SPLIT)
            )
            self.bev_path = self.bev_path / 'train'
        elif phase == "verifying":
            phase_scenes = CUSTOM_SPLIT
            self.bev_path = self.bev_path / 'val'
        
        # create a list of camera & lidar scans
        for scene_idx in range(len(self.nusc.scene)):
            scene = self.nusc.scene[scene_idx]
            if scene["name"] in phase_scenes:
                self.create_list_of_scans(scene)

    def __len__(self):
        return len(self.frame_list)

    def load_bev_map(self, lidar_token):
        bev_map_name = 'bev_map_' + lidar_token + '.npz'
        return np.load(str(self.bev_path / bev_map_name))['arr_0']
    
    def __getitem__(self, idx):
        # TODO
        return_dict = dict()
        data = self.frame_list[idx]['data']
        lidar_token = data['LIDAR_TOP']
        bev_map = self.load_bev_map(lidar_token)
        return_dict["bev_map"] = bev_map
        return_dict["lidar_token"] = lidar_token
        
        return return_dict

# TODO
def collate_box_point_fn(list_data):
    batch = {}
    for key in list_data[0]:
        batch[key] = [l[key] for l in list_data]
    
    return batch

def make_dataset(config):
    dataset = config.DATASET
    # Dataset
    if dataset.NAME.lower() == "nuscenes":
        Dataset = NuscenesBEVDataset
    else:
        raise Exception("Dataset Unknown")
    
    if dataset.DATA_SPLIT['train'] in ("parametrize", "parametrizing"):
        phase_train = "parametrizing"
        phase_val = "verifying"
    else:
        phase_train = "train"
        phase_val = "val"

    # Train dataset
    train_dataset = Dataset(
        phase=phase_train,
        config=config,
        shuffle=False,
    )
    print("Dataset Loaded")
    print("training size: ", len(train_dataset))

    # Val dataset
    if dataset.NAME.lower() == "nuscenes":
        val_dataset = Dataset(
            phase=phase_val,
            config=config,
            shuffle=False,
            cached_nuscenes=train_dataset.nusc,
        )
    print("validation size: ", len(val_dataset))

    return train_dataset, val_dataset

# TODO
def generate_box_point_from_bev(config, logger, dataloader, mode=None):

    bev_map_size = int((config.DATASET.POINT_CLOUD_RANGE[3] - config.DATASET.POINT_CLOUD_RANGE[0]) / (config.DATASET.VOXEL_SIZE[0] * config.MODEL.BEV_GRID_SIZE))
    text_categories = config.DATASET.TEXT_CATEGORIES

    logger.info(f"==============Generate box and point of {mode} dataset==============")

    count_box_point = 0

    bev_box_path = config.DATASET.BOX_SAVE_PATH / mode
    bev_box_path.mkdir(parents=True, exist_ok=True)
    bev_point_path = config.DATASET.POINT_SAVE_PATH / mode
    bev_point_path.mkdir(parents=True, exist_ok=True)

    data_iter = iter(dataloader)

    for i in range(len(dataloader)):
        batch = next(data_iter)

        bev_map = batch["bev_map"]
        lidar_token = batch["lidar_token"]

        for j in range(len(lidar_token)):
            bev_box = {}
            bev_point = {}
            sample = bev_map[j]
            count = 0
            for c in range(text_categories):
                sample_c = np.argwhere(sample == c)
                if len(sample_c) >= config.MIN_SAMPLES:
                    # TODO
                    min_samples = config.MIN_SAMPLES
                    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_samples, min_samples=min_samples)
                    cluster_labels = clusterer.fit_predict(sample_c)
                    
                    unique_labels = np.unique(cluster_labels)

                    boxes = []
                    points = []
                    for label in unique_labels:
                        if label != -1:
                            cluster_points = sample_c[cluster_labels == label]

                            x_min, y_min = np.min(cluster_points, axis=0)
                            x_max, y_max = np.max(cluster_points, axis=0)

                            if config.FILTER:
                                if (x_max - x_min) > config.LARGE_CLUSTER_SIZE or (y_max - y_min) > config.LARGE_CLUSTER_SIZE:
                                    continue

                            box = [[y_min, x_min], [y_max, x_max]]
                            boxes.append(box)
                            
                            point_x, point_y = random.choice(cluster_points)
                            point = [point_y, point_x]
                            points.append(point)

                    count += len(boxes)
                    bev_box[c] = boxes
                    bev_point[c] = points
                    # logger.info(f'Number of cluster in category {c}:{len(boxes)}   Number of pc:{len(sample_c)}   min_cluster_size:{min_cluster_size}')
                else:
                    bev_box[c] = []
                    bev_point[c] = []

            # assert count > 0
            
            count_box_point += 1

            bev_box_filename = 'bev_box_' + lidar_token[j] + '.pkl'
            with open(str(bev_box_path / bev_box_filename), 'wb') as file:
                pickle.dump(bev_box, file)
            logger.info(f'------Generate {mode} dataset bev box No.{count_box_point}:{bev_box_filename} total count:{count}------')

            bev_point_filename = 'bev_point_' + lidar_token[j] + '.pkl'
            with open(str(bev_point_path / bev_point_filename), 'wb') as file:
                pickle.dump(bev_point, file)
            logger.info(f'------Generate {mode} dataset bev point No.{count_box_point}:{bev_point_filename} total count:{count}------')


def parse_config():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument("--config_file", type=str, default='/opt/data/private/CSBEV/utils/config/generate_bev_box_mini.yaml', help="specify the config for processing point to bev")
    parser.add_argument("--bp_save_path", type=str, default=None, help="provide a path to save the generated BOX and POINT")
    parser.add_argument("--bev_name", type=str, default='ng1_lb-1', help="bev name")
    parser.add_argument("--train", action='store_true', default=False, help='generate train dataset')
    parser.add_argument("--val", action='store_true', default=False, help='generate val dataset')
    
    parser.add_argument("--min_samples", type=int, default=3, help='HDBSCAN: min_samples')
    parser.add_argument("--filter", action='store_true', default=False, help='Whether to filter large clusters')
    parser.add_argument("--large_cluster_size", type=int, default=10, help='true large cluster size = large_cluster_size * BEV_GRID_SIZE * VOXEL_SIZE')
    
    args = parser.parse_args()
    
    config = generate_config(args.config_file)

    if args.bp_save_path:
        config.DATASET.BOX_SAVE_PATH = Path(args.bp_save_path)
        config.DATASET.POINT_SAVE_PATH = Path(args.bp_save_path)
    else:
        folder_name = args.bev_name + '_ms' + str(args.min_samples) + '_f' + str(args.large_cluster_size)
        config.DATASET.BOX_SAVE_PATH = Path(config.DATASET.BOX_SAVE_PATH, 'box', folder_name)
        config.DATASET.POINT_SAVE_PATH = Path(config.DATASET.POINT_SAVE_PATH, 'point', folder_name)

    config.DATASET.BOX_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    config.DATASET.POINT_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    config.BEV_NAME = args.bev_name
    config.TRAIN = args.train
    config.VAL = args.val

    config.MIN_SAMPLES = args.min_samples
    config.FILTER = args.filter
    config.LARGE_CLUSTER_SIZE = args.large_cluster_size
    return config

def main():
    config = parse_config()

    log_file = Path(config.DATASET.LOG_SAVE_PATH, 'log_box_point_preprocess.txt')
    if os.path.isfile(str(log_file)):
        os.remove(str(log_file))
    logger = make_logger(log_file, 0)
    logger.info("==============Logging config==============")
    log_config(config, logger)

    logger.info("==============Start make dataset==============")
    train_dataset, val_dataset = make_dataset(config)
    logger.info("==============Start make dataloader==============")
    train_dataloader, val_dataloader = make_dataloader(config, train_dataset, val_dataset, collate_box_point_fn)

    logger.info("==============Start generate_box_point==============")
    if config.TRAIN:
        generate_box_point_from_bev(config, logger, train_dataloader, mode='train')
    if config.VAL:
        generate_box_point_from_bev(config, logger, val_dataloader, mode='val')

if __name__ == "__main__":
    main()