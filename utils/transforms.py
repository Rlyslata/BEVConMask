import torch
import numpy as np
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms.functional import resized_crop

def revtrans_rotation(pc, trans_dict):
    angle = np.random.random() * 2 * np.pi
    c = np.cos(angle)
    s = np.sin(angle)
    rotation = np.array(
        [[c, -s], [s, c]], dtype=np.float32
    )
    pc[:, :2] = pc[:, :2] @ rotation
    rotation = np.pad(rotation, (0, 1))
    rotation[2, 2] = 1.
    trans_dict['R'] = rotation.T
    return pc, trans_dict

def revtrans_translation(pc, trans_dict):
    translation = np.clip(np.random.normal(size=2, scale=4.).astype(np.float32), -15, 15)  # no trans along z
    pc[:, :2] += translation
    trans_dict['T'] = np.pad(translation, (0, 1))
    return pc, trans_dict


def resized_crop_adapt_maskclip(images, pairing_points, pairing_images):
    crop_size=(224, 416)
    crop_range=[0.3, 1.0]
    crop_ratio=(14.0 / 9.0, 17.0 / 9.0)
    imgs = torch.empty(
            (images.shape[0], 3) + tuple(crop_size), dtype=torch.float32
        )
    pairing_points_out = np.empty(0, dtype=np.int32)
    pairing_images_out = np.empty((0, 3), dtype=np.int32)
    for id, img in enumerate(images):
        successfull = False
        mask = pairing_images[:, 0] == id
        P1 = pairing_points[mask]
        P2 = pairing_images[mask]
        while not successfull:
            i, j, h, w = RandomResizedCrop.get_params(
                img, crop_range, crop_ratio
            )
            p1 = P1.copy()
            p2 = P2.copy()
            p2 = np.round(
                np.multiply(
                    p2 - [0, i, j],
                    [1.0, crop_size[0] / h, crop_size[1] / w],
                )
            ).astype(np.int32)

            valid_indexes_0 = np.logical_and(
                p2[:, 1] < crop_size[0], p2[:, 1] >= 0
            )
            valid_indexes_1 = np.logical_and(
                p2[:, 2] < crop_size[1], p2[:, 2] >= 0
            )
            valid_indexes = np.logical_and(valid_indexes_0, valid_indexes_1)
            sum_indexes = valid_indexes.sum()
            len_indexes = len(valid_indexes)
            if sum_indexes > 1024 or sum_indexes / len_indexes > 0.75:
                successfull = True
        imgs[id] = resized_crop(
            img, i, j, h, w, crop_size
        )
        pairing_points_out = np.concatenate(
            (pairing_points_out, p1[valid_indexes])
        )
        pairing_images_out = np.concatenate(
            (pairing_images_out, p2[valid_indexes])
        )
    
    return imgs, pairing_points_out, pairing_images_out