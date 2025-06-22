import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn.functional as F
import cumm.tensorview as tv
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points
from utils.config import generate_config
from pretrain.model_builder import make_model
from pretrain.dataloader_nuscenes import CollateSpconv
from spconv.utils import Point2VoxelCPU3d as VoxelGenerator

def get_xylims(dataset):
    if dataset == 'nuscenes':
        xlim = [0, 256]
        ylim = [0, 256]
    else:
        raise Exception('Unknown dataset')
    return xlim, ylim

def draw_box(ax, lidar_token, class_list, box_folder_name):
    box_path = '/opt/data/private/dataset/nuscenes-mini/box/' + box_folder_name + '/train/'
    box_file_name = 'bev_box_' + lidar_token + '.pkl'
    with open(box_path+box_file_name, 'rb') as f:
        box_dict = pickle.load(f)
        
    for key in class_list:
        boxes = box_dict[key]
        print(f'{key} : {boxes}')
        for i in range(len(boxes)):
            rect = patches.Rectangle((boxes[i][0][0], 255-boxes[i][1][1]), 
                                     boxes[i][1][0] - boxes[i][0][0], 
                                     boxes[i][1][1] - boxes[i][0][1], 
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

def draw_point(ax, lidar_token, class_list, point_folder_name):
    point_path = '/opt/data/private/dataset/nuscenes-mini/point/' + point_folder_name + '/train/'
    point_file_name = 'bev_point_' + lidar_token + '.pkl'
    with open(point_path+point_file_name, 'rb') as f:
        point_dict = pickle.load(f)
        
    for key in class_list:
        points = point_dict[key]
        print(f'{key} : {points}')
        for i in range(len(points)):
            ax.scatter(points[i][0], 255-points[i][1], s=500, c='green', marker="*", edgecolor="white", linewidth=1.25)

def draw_mask(lidar_token, class_list, bev_folder_name):
    import cv2
    bev_path = '/opt/data/private/dataset/nuscenes-mini/bev/' + bev_folder_name + '/train/'
    bev_map_name = 'bev_map_' + lidar_token + '.npz'
    bev_map = np.load(bev_path + bev_map_name)['arr_0']

    for key in class_list:
        bev_mask = (bev_map == key)
        bev_mask = bev_mask.astype(np.uint8) * 255
        cv2.imwrite('/opt/data/private/CSBEV/picture/bev_mask_' + str(key) + '.png', bev_mask)

def get_color(key):
    colors = np.array([[255/255, 105/255, 180/255, 0.8], # Hotpink
                       [255/255, 140/255,   0/255, 0.8], # Darkorange
                       [135/255, 206/255, 235/255, 0.8], # Lightskyblue
                       [255/255, 255/255,   0/255, 0.8], # Yellow
                       [  0/255, 255/255, 127/255, 0.8], # Springgreen
                       [123/255, 104/255, 238/255, 0.8]])# Mediumslateblue
    return colors[key]

def draw_bev(ax, lidar_token
             , class_list
             , bev_folder_name
             , bev_mask=None):
    bev_path = '/opt/data/private/dataset/nuscenes-mini/bev/' + bev_folder_name + '/train/'
    bev_map_name = 'bev_map_' + lidar_token + '.npz'
    bev_map = np.load(bev_path + bev_map_name)['arr_0']

    if bev_mask is not None:
        bev_map = bev_map * (bev_mask == -1) + bev_mask * (bev_mask != -1)

    for key in class_list:
        color = get_color(key)
        mask = np.flip((bev_map == key), axis=0)
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

def draw_rect(ax, corners, color):
    prev = corners[-1]
    for corner in corners:
        ax.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=2)
        prev = corner

def draw_pc(pc, lidar_token=None
            , lower_bound=-2
            , show_bev=False
            , show_gt=False
            , show_box=False
            , show_point=False
            , class_list=None
            , save_path=None
            , bev_folder_name=None
            , box_folder_name=None
            , point_folder_name=None
            , bev_mask=None
            , boxes=None):
    pc = pc[(pc[:, 2] >= lower_bound) & (pc[:, 2] <= 1)]
    pc = (pc[:, :2] + 51.2)*10/4
    fig, ax = plt.subplots(figsize=(40,40))
    xlim, ylim = get_xylims('nuscenes')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.scatter(pc[:,0], pc[:,1], s=2.5, c='m')

    if show_bev:
        draw_bev(ax, lidar_token, class_list, bev_folder_name, bev_mask=bev_mask)

    if show_gt:
        for box in boxes:
            bottom_corner = box.bottom_corners()[:2, :].T
            bottom_corner = (bottom_corner + 51.2)*10/4
            draw_rect(ax, bottom_corner, 'r')
        
    if show_box:
        draw_box(ax, lidar_token, class_list, box_folder_name)

    if show_point:
        draw_point(ax, lidar_token, class_list, point_folder_name)

    if save_path is not None:
        plt.savefig(save_path)
    
    plt.close()

def pc(scene_id):
    dataset_root = '/opt/data/private/dataset/nuscenes-mini'
    nusc = NuScenes(version="v1.0-mini", dataroot=dataset_root, verbose=False)

    scene = nusc.scene[scene_id]
    sample_token = scene['first_sample_token']
    sample = nusc.get('sample', sample_token)

    lidar_token = sample['data']['LIDAR_TOP']  
    lidar_data_path = nusc.get_sample_data_path(lidar_token)
    pc = LidarPointCloud.from_file(lidar_data_path)
    
    # pc
    # draw_pc(pc.points.T, lidar_token=lidar_token, lower_bound=-3, save_path="/opt/data/private/CSBEV/picture/pc3-" + str(scene_id) + ".png")
    # draw_pc(pc.points.T, lidar_token=lidar_token, lower_bound=-2, save_path="/opt/data/private/CSBEV/picture/pc2-" + str(scene_id) + ".png")
    # draw_pc(pc.points.T, lidar_token=lidar_token, lower_bound=-1, save_path="/opt/data/private/CSBEV/picture/pc1-" + str(scene_id) + ".png")
    draw_pc(pc.points.T, lidar_token=lidar_token, save_path="/opt/data/private/CSBEV/picture/pc-pink-" + str(scene_id) + ".png")

def bev_grid(scene_id, class_list, lower_bound=-2, bev_folder_name='ng1_lb-2.0'):
    dataset_root = '/opt/data/private/dataset/nuscenes-mini'
    nusc = NuScenes(version="v1.0-mini", dataroot=dataset_root, verbose=False)

    scene = nusc.scene[scene_id]
    sample_token = scene['first_sample_token']
    sample = nusc.get('sample', sample_token)

    lidar_token = sample['data']['LIDAR_TOP']  
    lidar_data_path = nusc.get_sample_data_path(lidar_token)
    pc = LidarPointCloud.from_file(lidar_data_path)

    # BEVGrid
    draw_pc(pc.points.T, lidar_token=lidar_token, show_bev=True, class_list=class_list, lower_bound=lower_bound, save_path="/opt/data/private/CSBEV/picture/bevgrid-" + str(scene_id) + "-" + bev_folder_name + ".png", bev_folder_name=bev_folder_name)

def bev_mask(scene_id, class_list, conv):
    dataset_root = '/opt/data/private/dataset/nuscenes-mini'
    nusc = NuScenes(version="v1.0-mini", dataroot=dataset_root, verbose=False)

    scene = nusc.scene[scene_id]
    sample_token = scene['first_sample_token']
    sample = nusc.get('sample', sample_token)

    lidar_token = sample['data']['LIDAR_TOP']  
    lidar_data_path = nusc.get_sample_data_path(lidar_token)
    pc = LidarPointCloud.from_file(lidar_data_path)

    # BEVMask-1/3
    config = generate_config('/opt/data/private/CSBEV/config/text_point_point/tpp_tp_bm_box_pp_bc.yaml')
    config.CONV = conv
    # TODO
    if conv==1:
        pretrain_path='/opt/data/private/result/pretrain/bm_pb_bc-1/model_epoch20.pt'
    if conv==3:
        pretrain_path='/opt/data/private/result/pretrain/bm_pb_bc-2/model_epoch20.pt'
    model_points, model_SAM, model_adapt_SAM = make_model(config)
    # load pretrain params
    pretrain_checkpoint = torch.load(pretrain_path, map_location=torch.device('cpu'))
    model_points.load_state_dict(pretrain_checkpoint['model_points'])
    model_adapt_SAM.load_state_dict(pretrain_checkpoint['model_adapt_SAM'])

    # transform pc to voxel
    points = np.fromfile(lidar_data_path, dtype=np.float32).reshape(-1, 5)[:, :4]
    points[:, 3] = points[:, 3] / 255
    points_mask = (points[:, 0] > -51.2) & (points[:, 0] < 51.2) \
        & (points[:, 1] > -51.2) & (points[:, 1] < 51.2)
    points = points[points_mask]

    voxel_generator = VoxelGenerator(
        vsize_xyz=[0.05, 0.05, 0.1],
        coors_range_xyz=[-51.2, -51.2, -3.0, 51.2, 51.2, 1.0],
        num_point_features=4,
        max_num_points_per_voxel=10,
        max_num_voxels=60000
    )
    voxel_output = voxel_generator.point_to_voxel(tv.from_numpy(points))
    tv_voxels, tv_coordinates, tv_num_points = voxel_output
    voxels = tv_voxels.numpy()
    coordinates = tv_coordinates.numpy()
    num_points = tv_num_points.numpy()

    coordinates = torch.from_numpy(coordinates)
    points_mean = torch.from_numpy(voxels).sum(dim=1, keepdim=False)
    normalizer = torch.clamp_min(torch.from_numpy(num_points).view(-1, 1), min=1.0).type_as(points_mean)
    points_mean = points_mean / normalizer
    voxels = points_mean.contiguous()

    voxels_in = []
    voxels_in.append(voxels)
    voxels_in = torch.cat(voxels_in)

    coordinates_in = []
    coords = []
    coords.append(F.pad(coordinates, (1, 0, 0, 0), value=0))
    coordinates_in = torch.cat(coords, 0).int()

    # prompt
    box_folder_name = 'ng1_lb-1_ms3_f10'
    box_path = '/opt/data/private/dataset/nuscenes-mini/box/' + box_folder_name + '/train/'
    box_file_name = 'bev_box_' + lidar_token + '.pkl'
    with open(box_path+box_file_name, 'rb') as f:
        box_dict = pickle.load(f)
    key = class_list[0]
    # 1*n*2*2
    boxes = box_dict[key]
    boxes = torch.stack([torch.from_numpy(np.stack(boxes))], axis=0)
    # 1*n*2
    boxes_label = (torch.ones(boxes.shape[0], boxes.shape[1], 2) * torch.tensor([2, 3])).int()

    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_points = model_points.to(device)
    model_SAM = model_SAM.to(device)
    model_adapt_SAM = model_adapt_SAM.to(device)
    model_points.eval()
    model_SAM.eval()
    model_adapt_SAM.eval()

    with torch.no_grad():
        bev_feature, _ = model_points(voxels_in.to(device), coordinates_in.to(device))
        bev_feature = model_adapt_SAM(bev_feature)
        bev_feature = F.normalize(bev_feature, p=2, dim=1)
        # sam_utils
        predicted_logits, predicted_iou = model_SAM(bev_feature, boxes.to(device), boxes_label.to(device),) 
        sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
        predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
        predicted_logits = torch.take_along_dim(
            predicted_logits, sorted_ids[..., None, None], dim=2
        )
        mask = torch.ge(predicted_logits[:, :, 0, :, :], 0)
        bev_mask, _ = torch.where(mask, torch.tensor(key).to(device), torch.tensor(-1).to(device)).max(dim=1)

    bev_mask = bev_mask.cpu().detach().numpy().squeeze()
    draw_pc(pc.points.T, lidar_token=lidar_token, show_bev=True, class_list=class_list, bev_mask=bev_mask, save_path="/opt/data/private/CSBEV/picture/bevmask" + str(conv) + "-" + str(scene_id) + ".png", bev_folder_name='ng1_lb-2.0')

def gt(scene_id):
    dataset_root = '/opt/data/private/dataset/nuscenes-mini'
    nusc = NuScenes(version="v1.0-mini", dataroot=dataset_root, verbose=False)

    scene = nusc.scene[scene_id]
    sample_token = scene['first_sample_token']
    sample = nusc.get('sample', sample_token)

    lidar_token = sample['data']['LIDAR_TOP']  
    lidar_data_path = nusc.get_sample_data_path(lidar_token)
    pc = LidarPointCloud.from_file(lidar_data_path)

    _, all_boxes, _ = nusc.get_sample_data(lidar_token)
    boxes = []
    for box in all_boxes:
        if box.name.startswith('vehicle'):
            boxes.append(box)

    draw_pc(pc.points.T, lidar_token=lidar_token, show_gt=True, boxes=boxes, save_path="/opt/data/private/CSBEV/picture/gt-" + str(scene_id) + ".png")

def draw_bev_grid_without_pc(scene_id, class_list, lower_bound, bev_folder_name):
    fig, ax = plt.subplots(figsize=(40,40))
    xlim, ylim = get_xylims('nuscenes')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    dataset_root = '/opt/data/private/dataset/nuscenes-mini'
    nusc = NuScenes(version="v1.0-mini", dataroot=dataset_root, verbose=False)

    scene = nusc.scene[scene_id]
    sample_token = scene['first_sample_token']
    sample = nusc.get('sample', sample_token)

    lidar_token = sample['data']['LIDAR_TOP']
    save_path="/opt/data/private/CSBEV/picture/bevgrid_without_pc-" + str(scene_id) + "-" + bev_folder_name + ".png"
    draw_bev(ax, lidar_token, class_list, bev_folder_name)
    plt.savefig(save_path)
    
    plt.close()

def main():
    scene_id = 0
    class_list = [0, 1, 2, 3, 4, 5]
    lower_bound = -2
    bev_folder_name = 'ng4_lb-2.0'
    pc(scene_id)
    # draw_bev_grid_without_pc(scene_id, class_list, lower_bound, bev_folder_name)
    # bev_grid(scene_id, class_list, lower_bound, bev_folder_name)
    # bev_mask(scene_id, class_list, 1)
    # bev_mask(scene_id, class_list, 3)
    # gt(scene_id)


if __name__ == "__main__":
    main()