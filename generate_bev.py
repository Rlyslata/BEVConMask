import argparse
from pathlib import Path
from utils.logger import make_logger
from utils.config import generate_config, log_config
from utils.transforms import resized_crop_adapt_maskclip
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

from model import (
    Preprocessing,
    maskClipFeatureExtractor,
)

CUSTOM_SPLIT = [
    "scene-0008", "scene-0009", "scene-0019", "scene-0029", "scene-0032", "scene-0042",
    "scene-0045", "scene-0049", "scene-0052", "scene-0054", "scene-0056", "scene-0066",
    "scene-0067", "scene-0073", "scene-0131", "scene-0152", "scene-0166", "scene-0168",
    "scene-0183", "scene-0190", "scene-0194", "scene-0208", "scene-0210", "scene-0211",
    "scene-0241", "scene-0243", "scene-0248", "scene-0259", "scene-0260", "scene-0261",
    "scene-0287", "scene-0292", "scene-0297", "scene-0305", "scene-0306", "scene-0350",
    "scene-0352", "scene-0358", "scene-0361", "scene-0365", "scene-0368", "scene-0377",
    "scene-0388", "scene-0391", "scene-0395", "scene-0413", "scene-0427", "scene-0428",
    "scene-0438", "scene-0444", "scene-0452", "scene-0453", "scene-0459", "scene-0463",
    "scene-0464", "scene-0475", "scene-0513", "scene-0533", "scene-0544", "scene-0575",
    "scene-0587", "scene-0589", "scene-0642", "scene-0652", "scene-0658", "scene-0669",
    "scene-0678", "scene-0687", "scene-0701", "scene-0703", "scene-0706", "scene-0710",
    "scene-0715", "scene-0726", "scene-0735", "scene-0740", "scene-0758", "scene-0786",
    "scene-0790", "scene-0804", "scene-0806", "scene-0847", "scene-0856", "scene-0868",
    "scene-0882", "scene-0897", "scene-0899", "scene-0976", "scene-0996", "scene-1012",
    "scene-1015", "scene-1016", "scene-1018", "scene-1020", "scene-1024", "scene-1044",
    "scene-1058", "scene-1094", "scene-1098", "scene-1107",
]


class NuscenesDataset(Dataset):
    def __init__(
        self,
        phase,
        config,
        **kwargs,
    ):
        self.phase = phase
        self.dataset_root = config.DATASET.DATASET_ROOT
        self.coors_range = config.DATASET.POINT_CLOUD_RANGE
        self.voxel_size = config.DATASET.VOXEL_SIZE
        self.bev_stride = config.MODEL.BEV_GRID_SIZE
        self.lower_bound = config.LOWER_BOUND
        self.num_grid = config.NUM_GRID

        self.corruptions = config.DATASET.get('CORRUPTIONS', 'None')
        self.severity = config.DATASET.get('SEVERITY', 1)
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
        elif phase == "verifying":
            phase_scenes = CUSTOM_SPLIT
        
        # create a list of camera & lidar scans
        for scene_idx in range(len(self.nusc.scene)):
            scene = self.nusc.scene[scene_idx]
            if scene["name"] in phase_scenes:
                self.create_list_of_scans(scene)

    def create_list_of_scans(self, scene):
        current_sample_token = scene["first_sample_token"]
        # Loop to get all successive keyframes
        sequence = []
        while current_sample_token != "":
            current_sample = self.nusc.get("sample", current_sample_token)
            sequence.append(current_sample)
            current_sample_token = current_sample["next"]

         # Add new scans in the list
        self.frame_list.extend(sequence)

    def load_point_cloud(self, sample):
        pointsensor = self.nusc.get("sample_data", sample["LIDAR_TOP"])
        pcl_path = osp.join(self.nusc.dataroot, pointsensor["filename"])
        points = np.fromfile(pcl_path, dtype=np.float32).reshape(-1, 5)[:, :4]
        points[:, 3] = points[:, 3] / 255
        cs_record = self.nusc.get(
            "calibrated_sensor", pointsensor["calibrated_sensor_token"]
        )
        Re = Quaternion(cs_record['rotation']).rotation_matrix.astype(np.float32)
        Te = np.array(cs_record["translation"], dtype=np.float32)
        poserecord = self.nusc.get("ego_pose", pointsensor["ego_pose_token"])
        Rw = Quaternion(poserecord['rotation']).rotation_matrix.astype(np.float32)
        Tw = np.array(poserecord["translation"], dtype=np.float32)
        R = Rw @ Re
        T = Rw @ Te + Tw
        return points, R, T

    def mask_points_by_range(self, points, limit_range):
        mask = (points[:, 0] > limit_range[0]) & (points[:, 0] < limit_range[3]) \
            & (points[:, 1] > limit_range[1]) & (points[:, 1] < limit_range[4]) \
            & (points[:, 2] >= self.lower_bound) & (points[:, 2] <= limit_range[5])
        return points[mask]
    def spatial_alignment_noise(points_xyz, severity):
        """
        Apply spatial misalignment noise to point cloud coordinates,
        simulating extrinsic calibration errors between LiDAR and camera.
        Noise levels follow the CVPR 2023 benchmark: 
        'Benchmarking Robustness of 3D Object Detection to Common Corruptions'.
        """
        # Translation noise standard deviation (meters)
        ct = [0.02, 0.04, 0.06, 0.08, 0.10][severity - 1] * 2
        # Rotation noise standard deviation (added directly to the 3×3 matrix)
        cr = [0.002, 0.004, 0.006, 0.008, 0.010][severity - 1] * 2

        r_noise = np.random.normal(size=(3, 3)) * cr
        t_noise = np.random.normal(size=3) * ct

        # Apply perturbation: x' = (I + r_noise) x + t_noise
        points_xyz = (np.eye(3) + r_noise) @ points_xyz + t_noise[:, np.newaxis]
        return points_xyz
    def map_pointcloud_to_image(self, point_merged, R, T, data, min_dist: float = 1.0):
        pc_original = LidarPointCloud.from_points(point_merged)

        images = []
        pairing_points = np.empty(0, dtype=np.int32)
        pairing_images = np.empty((0, 3), dtype=np.int32)
        camera_list = [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_FRONT_LEFT",
        ]
        
        if self.phase == 'train' and self.corruptions == 'spatial_alignment_noise':
            ct = [0.02, 0.04, 0.06, 0.08, 0.10][self.severity - 1] * 2
            cr = [0.002, 0.004, 0.006, 0.008, 0.010][self.severity - 1] * 2
            r_noise = np.random.normal(size=(3, 3)) * cr
            t_noise = np.random.normal(size=3) * ct
            noise_matrix = np.eye(3) + r_noise
        else:
            noise_matrix = np.eye(3)
            t_noise = np.zeros(3)
            
        for i, camera_name in enumerate(camera_list):
            pc = copy.deepcopy(pc_original)
            cam = self.nusc.get("sample_data", data[camera_name])
            im = np.array(Image.open(os.path.join(self.nusc.dataroot, cam["filename"])))

            # Points live in the point sensor frame. So they need to be transformed via
            # global to the image plane.
            # First step: transform the pointcloud to the ego vehicle frame for the
            # timestamp of the sweep.
            # cs_record = self.nusc.get(
            #     "calibrated_sensor", pointsensor["calibrated_sensor_token"]
            # )
            # pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
            # pc.translate(np.array(cs_record["translation"]))

            # Second step: transform from ego to the global frame.
            # poserecord = self.nusc.get("ego_pose", pointsensor["ego_pose_token"])
            # pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix)
            # pc.translate(np.array(poserecord["translation"]))
            pc.rotate(R)
            pc.translate(T)
           
            # Third step: transform from global into the ego vehicle frame for the
            # timestamp of the image.
            poserecord = self.nusc.get("ego_pose", cam["ego_pose_token"])
            pc.translate(-np.array(poserecord["translation"]))
            pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix.T)

            # Fourth step: transform from ego into the camera.
            cs_record = self.nusc.get(
                "calibrated_sensor", cam["calibrated_sensor_token"]
            )
            pc.translate(-np.array(cs_record["translation"]))
            pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix.T)

            # ---------- spatial misalignment noise----------
            if self.phase == 'train' and self.corruptions == 'spatial_alignment_noise':
                pc.points[:3, :] = noise_matrix @ pc.points[:3, :] + t_noise[:, np.newaxis]
            # ---------------------------------------
            
            # Fifth step: actually take a "picture" of the point cloud.
            # Grab the depths (camera frame z axis points away from the camera).
            depths = pc.points[2, :]

            # Take the actual picture
            # (matrix multiplication with camera-matrix + renormalization).
            points = view_points(
                pc.points[:3, :],
                np.array(cs_record["camera_intrinsic"]),
                normalize=True,
            )

            # Remove points that are either outside or behind the camera.
            # Also make sure points are at least 1m in front of the camera to avoid
            # seeing the lidar points on the camera
            # casing for non-keyframes which are slightly out of sync.
            points = points[:2].T
            mask = np.ones(depths.shape[0], dtype=bool)
            mask = np.logical_and(mask, depths > min_dist)
            mask = np.logical_and(mask, points[:, 0] > 0)
            mask = np.logical_and(mask, points[:, 0] < im.shape[1] - 1)
            mask = np.logical_and(mask, points[:, 1] > 0)
            mask = np.logical_and(mask, points[:, 1] < im.shape[0] - 1)

            # point index
            matching_points = np.where(mask)[0]
            # pixel index
            matching_pixels = np.round(
                np.flip(points[matching_points], axis=1)
            ).astype(np.int32)
            images.append(im / 255)
            # Matches all paired point-pixel pairs
            pairing_points = np.concatenate((pairing_points, matching_points))
            # Structure: image index + pixel index
            pairing_images = np.concatenate(
                (
                    pairing_images,
                    np.concatenate(
                        (
                            np.ones((matching_pixels.shape[0], 1), dtype=np.int32) * i,
                            matching_pixels,
                        ),
                        axis=1,
                    ),
                )
            )

        return images, pairing_points, pairing_images

    def __len__(self):
        return len(self.frame_list)

    def __getitem__(self, idx):
        return_dict = dict()

        pc, R, T = self.load_point_cloud(self.frame_list[idx]['data'])
        pc = self.mask_points_by_range(pc, self.coors_range)

        # pair image-point cloud
        data = self.frame_list[idx]['data']
        lidar_token = data['LIDAR_TOP']
        (
            images,
            pairing_points,
            pairing_images,
        ) = self.map_pointcloud_to_image(pc, R, T, data)

        images = torch.tensor(np.array(images, dtype=np.float32).transpose(0, 3, 1, 2))

        images, pairing_points, pairing_images = resized_crop_adapt_maskclip(images, pairing_points, pairing_images)
        
        bev_grid_size = self.bev_stride * self.voxel_size[0] * self.num_grid
        pairing_points_xy = pc[:, :2][pairing_points]
        bev_index = np.floor((pairing_points_xy + self.coors_range[3]) / bev_grid_size).astype(np.int32)
        assert np.all(bev_index < 256/self.num_grid)

        return_dict["images_out"] = images
        return_dict["bev_index"] = bev_index
        return_dict["pairing_images"] = pairing_images
        return_dict["lidar_token"] = lidar_token

        return return_dict


def make_dataset(config):
    dataset = config.DATASET
    # Dataset
    if dataset.NAME.lower() == "nuscenes":
        Dataset = NuscenesDataset
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


def collate_bev_fn(list_data):
    batch = {}
    for key in list_data[0]:
        batch[key] = [l[key] for l in list_data]

    # return_dict["images_out"] = images
    # return_dict["bev_index"] = bev_index
    # return_dict["pairing_images"] = pairing_images
    # return_dict["lidar_token"] = lidar_token
    offset = 0
    batch["offset"] = []
    for batch_id in range(len(batch["bev_index"])):
        batch["pairing_images"][batch_id][:, 0] += batch_id * batch["images_out"][0].shape[0]
        offset += len(batch["bev_index"][batch_id])
        batch['offset'].append(offset)

    batch['images'] = torch.cat(batch['images_out'], 0)
    batch['pairing_images'] = torch.tensor(np.concatenate(batch['pairing_images'], axis=0))
    batch["bev_index"] = torch.tensor(np.concatenate(batch["bev_index"], axis=0))
    return batch


def make_dataloader(config, train_dataset, val_dataset, collate_fn):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.OPTIMIZATION.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=config.OPTIMIZATION.NUM_WORKERS_PER_GPU,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False,
        worker_init_fn=lambda id: np.random.seed(
            torch.initial_seed() // 2 ** 32 + id
        ),
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.OPTIMIZATION.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=config.OPTIMIZATION.NUM_WORKERS_PER_GPU,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False,
        worker_init_fn=lambda id: np.random.seed(
            torch.initial_seed() // 2 ** 32 + id
        ),
    )

    return train_dataloader, val_dataloader


def make_model(config):
    if config.MODEL.IMAGE_ENCODER:
        if config.IMAGE_ENCODER.NAME.lower() == "maskclip":
            model_images = maskClipFeatureExtractor(config, preprocessing=Preprocessing())
        else:
            raise Exception(f"Images model not found: {config.IMAGE_ENCODER.NAME}")
    else:
        raise Exception(f"Image encoder not enabled")
    return model_images


def generate_bev(config, logger, model_images, dataloader, device, mode=None):
    
    model_images.eval()

    bev_map_size = int((config.DATASET.POINT_CLOUD_RANGE[3] - config.DATASET.POINT_CLOUD_RANGE[0]) / (config.DATASET.VOXEL_SIZE[0] * config.MODEL.BEV_GRID_SIZE * config.NUM_GRID))
    text_categories = config.DATASET.TEXT_CATEGORIES

    logger.info(f"==============Generate bev of {mode} dataset==============")

    count_bev = 0

    bev_map_path = config.DATASET.BEV_SAVE_PATH / mode
    bev_map_path.mkdir(parents=True, exist_ok=True)

    data_iter = iter(dataloader)

    for i in range(len(dataloader)):
        batch = next(data_iter)

        images = batch["images"].to(device)
        bev_index = batch["bev_index"]
        pairing_images = batch["pairing_images"].to(device)
        lidar_token = batch["lidar_token"]
        offset = batch["offset"]
        
        with torch.no_grad():
            _, images_pred = model_images(images)

        pairing_images_T = tuple(pairing_images.T.long())
        images_pred = images_pred[pairing_images_T]
        images_pred = images_pred.cpu()

        index = 0
        for j in range(len(lidar_token)):
            sample_original = torch.cat((bev_index[index:offset[j]], images_pred[index:offset[j]].unsqueeze(1)), dim=1)
            index = offset[j]
            # TODO 
            sample = torch.zeros(sample_original.shape[0], 3, dtype=torch.int)
            sample[:, 0] = bev_map_size - 1 - sample_original[:, 1]
            sample[:, 1] = sample_original[:, 0]
            sample[:, 2] = sample_original[:, 2]
            sample = sample.numpy()
            
            # count the number of occurrences of [x, y, c]
            count_matrix = np.bincount(sample[:, 0]*bev_map_size*text_categories + sample[:, 1]*text_categories + sample[:, 2], 
                                    minlength=bev_map_size*bev_map_size*text_categories
                                    ).reshape(bev_map_size, bev_map_size, text_categories)
            
            # find the c value that occurs most often at the [x, y] position and store it in bev_map
            argmax_result = count_matrix.argmax(axis=2)

            max_occurrences = count_matrix.max(axis=2)
            total_occurrences = count_matrix.sum(axis=2)
            total_occurrences[total_occurrences == 0] = 1
            threshold_percentage = 0.5
            mask = (max_occurrences / total_occurrences >= threshold_percentage) & (max_occurrences >= 5)

            # mask = count_matrix.max(axis=2) >= 5
            bev_map = np.where(mask, argmax_result, -1)
            if config.NUM_GRID>1:
                bev_map = np.repeat(bev_map, repeats=config.NUM_GRID, axis=0)
                bev_map = np.repeat(bev_map, repeats=config.NUM_GRID, axis=1)

            # save
            bev_map_filename = 'bev_map_' + lidar_token[j] + '.npz'
            # torch.save(bev_map, str(bev_map_path / bev_map_filename))
            np.savez_compressed(str(bev_map_path / bev_map_filename), bev_map)
            count_bev += 1
            logger.info(f'------Generate {mode} dataset bev map No.{count_bev} : {bev_map_filename}------')


def parse_config():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument("--config_file", type=str, default='/opt/data/private/CSBEV/utils/config/generate_bev_box_mini.yaml', help="specify the config for processing point to bev")
    parser.add_argument("--bev_save_path", type=str, default=None, help="provide a path to save the generated BEV")
    parser.add_argument("--lower_bound", type=float, default=-1, help='lower bound for filtering ground point clouds')
    parser.add_argument("--num_grid", type=int, default=1, help='true grid size = num_grid * BEV_GRID_SIZE * VOXEL_SIZE')
    parser.add_argument("--train", action='store_true', default=False, help='generate train dataset')
    parser.add_argument("--val", action='store_true', default=False, help='generate val dataset')
    args = parser.parse_args()
    
    config = generate_config(args.config_file)

    if args.bev_save_path:
        config.DATASET.BEV_SAVE_PATH = Path(args.bev_save_path)
    else:
        folder_name = 'ng' + str(args.num_grid) + '_lb' + str(args.lower_bound)
        config.DATASET.BEV_SAVE_PATH = Path(config.DATASET.BEV_SAVE_PATH, 'bev', folder_name)

    config.DATASET.BEV_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    config.LOWER_BOUND = args.lower_bound
    config.NUM_GRID = args.num_grid
    config.TRAIN = args.train
    config.VAL = args.val
    return config


def main():
    config = parse_config()

    log_file = Path(config.DATASET.LOG_SAVE_PATH, 'log_bev_preprocess.txt')
    if os.path.isfile(str(log_file)):
        os.remove(str(log_file))
    logger = make_logger(log_file, 0)
    logger.info("==============Logging config==============")
    log_config(config, logger)

    logger.info("==============Start make dataset==============")
    train_dataset, val_dataset = make_dataset(config)
    logger.info("==============Start make dataloader==============")
    train_dataloader, val_dataloader = make_dataloader(config, train_dataset, val_dataset, collate_bev_fn)
    logger.info("==============Start make model==============")
    model_images = make_model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_images.to(device)
    if torch.cuda.device_count() > 1:
        model_images = DataParallel(model_images)

    logger.info("==============Start generate_bev==============")
    if config.TRAIN:
        generate_bev(config, logger, model_images, train_dataloader, device, mode='train')
    if config.VAL:
        generate_bev(config, logger, model_images, val_dataloader, device, mode='val')


if __name__ == "__main__":
    main()