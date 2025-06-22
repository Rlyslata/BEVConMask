import torch
import numpy as np
import pickle
import os.path as osp
import cumm.tensorview as tv
import torch.nn.functional as F
from pathlib import Path
from pyquaternion import Quaternion
from torch.utils.data import Dataset
from utils.transforms import revtrans_rotation, revtrans_translation
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from spconv.utils import Point2VoxelCPU3d as VoxelGenerator

# from memory_profiler import profile

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

class CollateSpconv:
    def __init__(self, config) -> None:

        self._voxel_generator = VoxelGenerator(
            vsize_xyz=config.DATASET.VOXEL_SIZE,
            coors_range_xyz=config.DATASET.POINT_CLOUD_RANGE,
            num_point_features=4,
            max_num_points_per_voxel=10,
            max_num_voxels=60000
        )
        self._voxel_generator_bev = VoxelGenerator(
            vsize_xyz=config.DATASET.VOXEL_SIZE_BEV,
            coors_range_xyz=config.DATASET.POINT_CLOUD_RANGE,
            num_point_features=4,
            max_num_points_per_voxel=config.DATASET.MAX_POINTS_PER_VOXEL_BEV,
            max_num_voxels=60000
        )

        self.load_bev = config.DATASET.BEV
        self.load_bev_mae = config.DATASET.BEV_MAE
        self.load_box_point = config.DATASET.BOX_POINT

    def generate(self, points, voxel_generator):
        voxel_output = voxel_generator.point_to_voxel(tv.from_numpy(points))
        tv_voxels, tv_coordinates, tv_num_points = voxel_output
        # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
        voxels = tv_voxels.numpy()
        coordinates = tv_coordinates.numpy()
        num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points
    
    # @profile(stream=open('/opt/data/private/memory/dataset_collate.log', 'w+'))
    def collate_spconv(self, list_data):
        batch = {}
        for key in list_data[0]:
            batch[key] = [l[key] for l in list_data]
        batch["batch_size"] = len(list_data)
        
        batch["voxels_out"] = []

        coords = []
        batch_id = 0
        for group_pc in batch["points_out"]:
            voxels, coordinates, num_points = self.generate(group_pc, self._voxel_generator)
            coordinates = torch.from_numpy(coordinates)
            points_mean = torch.from_numpy(voxels).sum(dim=1, keepdim=False)
            normalizer = torch.clamp_min(torch.from_numpy(num_points).view(-1, 1), min=1.0).type_as(points_mean)
            points_mean = points_mean / normalizer
            voxels = points_mean.contiguous()

            coords.append(F.pad(coordinates, (1, 0, 0, 0), value=batch_id))
            batch["voxels_out"].append(voxels)
            batch_id += 1
        batch['coordinates_out'] = torch.cat(coords, 0).int()
        batch['voxels_out'] = torch.cat(batch['voxels_out'])

        batch["voxels_bev"] = []
        batch["num_points_bev"] = []
        coords_bev = []
        batch_id = 0
        for group_pc in batch["points_out"]:
            voxels_bev, coordinates_bev, num_points_bev = self.generate(group_pc, self._voxel_generator_bev)
            voxels_bev = torch.from_numpy(voxels_bev)
            coordinates_bev = torch.from_numpy(coordinates_bev)
            num_points_bev = torch.from_numpy(num_points_bev)

            coords_bev.append(F.pad(coordinates_bev, (1, 0, 0, 0), value=batch_id))
            batch["voxels_bev"].append(voxels_bev)
            batch["num_points_bev"].append(num_points_bev)
            batch_id += 1
        batch['coordinates_bev'] = torch.cat(coords_bev, 0).int()
        batch['voxels_bev'] = torch.cat(batch['voxels_bev'])
        batch['num_points_bev'] = torch.cat(batch['num_points_bev'])

        batch['R_out'] = torch.stack([torch.from_numpy(np.stack(R)) for R in batch['R_out']], axis=0)
        batch['T_out'] = torch.stack([torch.from_numpy(np.stack(T)) for T in batch['T_out']], axis=0)
        if self.load_bev:
            batch['bev_map_out'] = torch.stack([torch.from_numpy(np.stack(bev_map)) for bev_map in batch['bev_map_out']], axis=0)
            batch['bev_mask_sem'] = torch.stack([torch.from_numpy(np.stack(bev_mask)) for bev_mask in batch['bev_mask_sem']], axis=0)
            batch['bev_mask_union'] = torch.stack([torch.from_numpy(np.stack(bev_mask)) for bev_mask in batch['bev_mask_union']], axis=0)
        if self.load_box_point:
            batch['box_key_out'] = torch.stack([torch.from_numpy(np.stack(box_key)) for box_key in batch['box_key_out']], axis=0)
            batch['box_value_out'] = torch.stack([torch.from_numpy(np.stack(box_value)) for box_value in batch['box_value_out']], axis=0)

            batch['point_key_out'] = torch.stack([torch.from_numpy(np.stack(point_key)) for point_key in batch['point_key_out']], axis=0)
            batch['point_value_out'] = torch.stack([torch.from_numpy(np.stack(point_value)) for point_value in batch['point_value_out']], axis=0)

        return batch

class NuscenesDatasetMAE(Dataset):

    def __init__(
        self,
        phase,
        config,
        **kwargs,
    ):
        self.phase = phase
        self.dataset_root = config.DATASET.DATASET_ROOT
        self.num_frames_in = config.DATASET.INPUT_FRAMES
        self.num_frames_out = config.DATASET.OUTPUT_FRAMES
        self.num_frames = self.num_frames_in + self.num_frames_out
        self.select = config.DATASET.SKIP_FRAMES + 1
        self.coors_range = config.DATASET.POINT_CLOUD_RANGE
        self.exc_road = config.DATASET.EXCEPT_ROAD

        self.load_bev = config.DATASET.BEV
        self.load_bev_mae = config.DATASET.BEV_MAE
        self.load_box_point = config.DATASET.BOX_POINT
        if self.load_bev:
            self.bev_path = Path(config.BEV_SAVE_PATH, 'bev', config.BEV_NAME)
            self.mask_ratio_semantic = config.DATASET.MASK_RATIO_SEMANTIC
            self.mask_ratio_non_semantic = config.DATASET.MASK_RATIO_NON_SEMANTIC
        if self.load_box_point:
            self.box_path = Path(config.BOX_SAVE_PATH, 'box', config.BOX_POINT_NAME)
            self.point_path = Path(config.POINT_SAVE_PATH, 'point', config.BOX_POINT_NAME)

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

        # a skip ratio can be used to reduce the dataset size and accelerate experiments
        try:
            skip_ratio = config.MODEL.DATASET_SKIP_STEP
        except KeyError:
            skip_ratio = 1
        skip_counter = 0
        
        if phase in ("train", "val", "test"):
            phase_scenes = create_splits_scenes()[phase]
        elif phase == "parametrizing":
            phase_scenes = list(
                set(create_splits_scenes()["train"]) - set(CUSTOM_SPLIT)
            )
            if self.load_bev:
                self.bev_path = self.bev_path / 'train'
            if self.load_box_point:
                self.box_path = self.box_path / 'train'
                self.point_path = self.point_path / 'train'
            if self.load_bev_mae:
                self.bev_mae_path = self.bev_mae_path / 'train'
        elif phase == "verifying":
            phase_scenes = CUSTOM_SPLIT
            if self.load_bev:
                self.bev_path = self.bev_path / 'val'
            if self.load_box_point:
                self.box_path = self.box_path / 'val'
                self.point_path = self.point_path / 'val'
            if self.load_bev_mae:
                self.bev_mae_path = self.bev_mae_path / 'val'

        # create a list of camera & lidar scans
        for scene_idx in range(len(self.nusc.scene)):
            scene = self.nusc.scene[scene_idx]
            if scene["name"] in phase_scenes:
                skip_counter += 1
                if skip_counter % skip_ratio == 0:
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
        for i in range(len(sequence) - self.num_frames * self.select + 1):
            self.frame_list.append([sequence[j] for j in range(i, i + self.num_frames * self.select, self.select)])

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
            & (points[:, 1] > limit_range[1]) & (points[:, 1] < limit_range[4])
        return points[mask]

    def load_bev_map(self, path, lidar_token):
        bev_map_name = 'bev_map_' + lidar_token + '.npz'
        return np.load(str(path / bev_map_name))['arr_0']
    
    def load_bev_box(self, lidar_token):
        bev_box_name = 'bev_box_' + lidar_token + '.pkl'
        with open(self.box_path / bev_box_name, 'rb') as f:
            return pickle.load(f)
        
    def load_bev_point(self, lidar_token):
        bev_point_name = 'bev_point_' + lidar_token + '.pkl'
        with open(self.point_path / bev_point_name, 'rb') as f:
            return pickle.load(f)

    def get_mask_by_bev_map(self, bev_map):
        # get mask
        bev_mask_mae_non_semantic = np.zeros_like(bev_map, dtype=bool)
        bev_mask_mae_semantic = np.zeros_like(bev_map, dtype=bool)

        if self.exc_road:
            negative_indices = np.where((bev_map == -1) | (bev_map == 2))
            non_negative_indices = np.where((bev_map != -1) & (bev_map != 2))
        else:
            negative_indices = np.where(bev_map == -1)
            non_negative_indices = np.where(bev_map != -1)

        mask_count_non_semantic = int(len(negative_indices[0]) * self.mask_ratio_non_semantic)
        mask_count_semantic = int(len(non_negative_indices[0]) * self.mask_ratio_semantic)

        random_indices_non_semantic = np.random.choice(len(negative_indices[0]), mask_count_non_semantic, replace=False)
        random_indices_semantic = np.random.choice(len(non_negative_indices[0]), mask_count_semantic, replace=False)

        bev_mask_mae_non_semantic[negative_indices[0][random_indices_non_semantic], negative_indices[1][random_indices_non_semantic]] = True
        bev_mask_mae_semantic[non_negative_indices[0][random_indices_semantic], non_negative_indices[1][random_indices_semantic]] = True
        
        return bev_mask_mae_semantic, bev_mask_mae_non_semantic

    def mask_pc_by_bev_map(self, pc, bev_map):
        # get mask
        bev_mask_mae_non_semantic = bev_map == -1
        bev_mask_mae_semantic = bev_map != -1

        # mask pc
        bev_grid_size = (self.coors_range[3] - self.coors_range[0]) / len(bev_map)
        bev_index = np.floor((pc[:, :2] + self.coors_range[3]) / bev_grid_size).astype(np.int32)
        x_indices, y_indices = bev_index.transpose()
        row_indices = len(bev_map) - 1 - y_indices
        col_indices = x_indices

        non_semantic_indices = bev_mask_mae_non_semantic[row_indices, col_indices]
        non_semantic_points = pc[non_semantic_indices]
        num_to_mask = int(self.mask_ratio_non_semantic * len(non_semantic_points))
        masked_indices = np.random.choice(len(non_semantic_points), len(non_semantic_points) - num_to_mask, replace=False)
        pc_non_semantic = non_semantic_points[masked_indices]

        semantic_indices = bev_mask_mae_semantic[row_indices, col_indices]
        semantic_points = pc[semantic_indices]
        num_to_mask = int(self.mask_ratio_semantic * len(semantic_points))
        masked_indices = np.random.choice(len(semantic_points), len(semantic_points) - num_to_mask, replace=False)
        pc_semantic = semantic_points[masked_indices]

        pc = np.concatenate((pc_non_semantic, pc_semantic))
        
        return pc
        

    def __len__(self):
        return len(self.frame_list)

    # @profile(stream=open('/opt/data/private/memory/dataset_getitem.log', 'w+'))
    def __getitem__(self, idx):
        return_dict = dict()

        pc, R, T = self.load_point_cloud(self.frame_list[idx][1]['data'])
        pc = self.mask_points_by_range(pc, self.coors_range)
        
        if self.load_bev:
            bev_map_out = self.load_bev_map(self.bev_path, self.frame_list[idx][1]['data']['LIDAR_TOP'])
            return_dict["bev_map_out"] = bev_map_out

            bev_mask_sem, bev_mask_non_sem = self.get_mask_by_bev_map( bev_map_out)
            bev_mask_union = np.logical_or(bev_mask_sem, bev_mask_non_sem)
            return_dict["bev_mask_sem"] = bev_mask_sem
            return_dict["bev_mask_union"] = bev_mask_union
        
        if self.load_bev_mae:
            bev_mae_map_out = self.load_bev_map(self.bev_mae_path, self.frame_list[idx][1]['data']['LIDAR_TOP'])
            return_dict["bev_mae_map_out"] = bev_mae_map_out

        if self.load_box_point:
            bev_box_out = self.load_bev_box(self.frame_list[idx][1]['data']['LIDAR_TOP'])
            box_with_key = [(key, element) for key, values in bev_box_out.items() for element in values if values]
            if not box_with_key:
                box_with_key.append((-1, [[0, 0], [0, 0]]))

            bev_point_out = self.load_bev_point(self.frame_list[idx][1]['data']['LIDAR_TOP'])
            point_with_key = [(key, element) for key, values in bev_point_out.items() for element in values if values]
            if not point_with_key:
                point_with_key.append((-1, [0, 0]))

            if len(box_with_key) >= 5:
                index = np.random.choice(np.arange(len(box_with_key)), 5, replace=False)
            else:
                index = np.random.choice(np.arange(len(box_with_key)), 5, replace=True)

            select_box_with_key = [box_with_key[i] for i in index]
            box_key, box_value = zip(*select_box_with_key)
            return_dict["box_key_out"] = box_key
            return_dict["box_value_out"] = box_value

            select_point_with_key = [point_with_key[i] for i in index]
            point_key, point_value = zip(*select_point_with_key)
            return_dict["point_key_out"] = point_key
            return_dict["point_value_out"] = point_value

        return_dict["points_out"] = pc
        return_dict["R_out"] = R
        return_dict["T_out"] = T

        return return_dict