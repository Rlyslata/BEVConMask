import os
import re
import random
import numpy as np
import torch
import copy
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
import spconv.pytorch as spconv
from utils import loss_utils
from pytorch_lightning.utilities import rank_zero_only

# from memory_profiler import profile

class LightningPretrain(pl.LightningModule):
    # @profile(stream=open('/opt/data/private/memory/train_init.log', 'w+'))
    def __init__(self, model_points, model_decoder, model_decoder_coor, model_decoder_sem, model_SAM, model_adapt_SAM, model_teacher, config):
        super().__init__()
        self.model_points = model_points
        self.config = config
        self.model_decoder = model_decoder
        self.model_decoder_coor = model_decoder_coor
        self.model_decoder_sem = model_decoder_sem
        self.model_SAM = model_SAM
        self.model_adapt_SAM = model_adapt_SAM
        self.model_teacher = model_teacher
        self.model_mask_token = nn.Parameter(torch.zeros(1, 4), requires_grad=True)

        self.range = config.DATASET.POINT_CLOUD_RANGE
        self.scale = self.range[3]
        self.input_frames = config.DATASET.INPUT_FRAMES

        self.model_losses = config.MODEL.LOSSES
        self.batch_size = config.OPTIMIZATION.BATCH_SIZE_PER_GPU
        self.num_epochs = config.OPTIMIZATION.NUM_EPOCHS

        self.exc_road = config.DATASET.EXCEPT_ROAD
        self.mask_with_sam = config.DATASET.MASK_WITH_SAM
        self.prompt = config.DATASET.PROMPT
        self.smooth = config.SMOOTH

        self.grid = 1
        self.down_factor = 8
        self.unshuffle = torch.nn.PixelUnshuffle(self.down_factor)
        self.mask_ratio_semantic = config.DATASET.MASK_RATIO_SEMANTIC
        self.mask_ratio_non_semantic = config.DATASET.MASK_RATIO_NON_SEMANTIC
        point_cloud_range = np.array(self.range)
        self.grid_size = np.round((point_cloud_range[3:6] - point_cloud_range[0:3]) / config.DATASET.VOXEL_SIZE).astype(np.int64)[::-1] + [1, 0, 0]
        voxel_size = config.DATASET.VOXEL_SIZE_BEV
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + self.range[0] 
        self.y_offset = self.vy / 2 + self.range[1]
        self.z_offset = self.range[2]

        self.coor_loss = loss_utils.MaskChamferDistance()
        self.CE = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.L1_loss = nn.L1Loss()
        
        self.epoch = 0
        if config.MODEL.RESUME_PATH is not None:
            self.epoch = int(re.search(r"(?<=epoch=)[0-9]+", config.MODEL.RESUME_PATH)[0]) + 1
            
        self.save_path = config.MODEL.SAVE_PATH
        if os.environ.get("LOCAL_RANK", 0) == 0:
            os.makedirs(self.save_path, exist_ok=True)

        self.saved = False if self.epoch < 20 else True

        if config.DATASET.TEXT:
            self.text_embeddings_path = config.DATASET.TEXT_EMBEDDINGS_PATH
            self.text_categories = config.DATASET.TEXT_CATEGORIES
            self.text_embeddings_dimension = config.DATASET.TEXT_EMBEDDINGS_DIMENSION
            self.register_buffer('text_embeddings', torch.randn(self.text_categories, self.text_embeddings_dimension))
            self.text_embeddings[:, :] = torch.load(self.text_embeddings_path, map_location='cuda')

    def configure_optimizers(self):
        optimizer_class = getattr(optim, self.config.OPTIMIZATION.OPTIMIZER)
        param_list = []
        param_list.extend(list(self.model_points.parameters()))
        if self.model_decoder is not None:
            param_list.extend(list(self.model_decoder.parameters()))
        if self.model_decoder_coor is not None:
            param_list.extend(list(self.model_decoder_coor.parameters()))
        if self.model_decoder_sem is not None:
            param_list.extend(list(self.model_decoder_sem.parameters()))
        if self.model_adapt_SAM is not None:
            param_list.extend(list(self.model_adapt_SAM.parameters()))
        optimizer = optimizer_class(
            param_list,
            lr=self.config.OPTIMIZATION.LR,
            weight_decay=self.config.OPTIMIZATION.WEIGHT_DECAY
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.num_epochs)
        # scheduler = LinearWarmupCosineAnnealingLR(optimizer, self.config.OPTIMIZATION.NUM_EPOCHS // 20, self.config.OPTIMIZATION.NUM_EPOCHS, self.config.OPTIMIZATION.LR, self.config.OPTIMIZATION.LR / 100)
        return [optimizer], [scheduler]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)
    
    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        """Create boolean mask by actually number of a padded tensor.

        Args:
            actual_num (torch.Tensor): Actual number of points in each voxel.
            max_num (int): Max number of points in each voxel

        Returns:
            torch.Tensor: Mask indicates which points are valid inside a voxel.
        """
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(
            max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def get_chamfer_and_gt_coor(self, batch):
        voxels_bev, coordinates_bev, num_points_bev = batch['voxels_bev'], batch['coordinates_bev'], batch['num_points_bev']
        f_center = torch.zeros_like(voxels_bev[:, :, :3])
        
        f_center[:, :, 0] = (voxels_bev[:, :, 0] - (coordinates_bev[:, 3].unsqueeze(dim=1) * self.vx + self.x_offset)) / self.vx
        f_center[:, :, 1] = (voxels_bev[:, :, 1] - (coordinates_bev[:, 2].unsqueeze(dim=1) * self.vy + self.y_offset)) / self.vy
        f_center[:, :, 2] = (voxels_bev[:, :, 2]) / self.vz
        voxel_count = f_center.shape[1]
        mask_num = self.get_paddings_indicator(num_points_bev, voxel_count, axis=0)
        mask_num = torch.unsqueeze(mask_num, -1).type_as(f_center)
        f_center *= mask_num

        sparse_shape = [1, self.grid_size[1]//self.down_factor, self.grid_size[2]//self.down_factor,]

        chamfer_mask = spconv.SparseConvTensor(
            mask_num.squeeze().detach(),
            coordinates_bev.int(),
            sparse_shape,
            self.batch_size
        ).dense()
        batch['chamfer_mask'] = chamfer_mask.sum(dim=2)
        n, m, _ = f_center.shape
        f_center = f_center.reshape(n, -1)

        pts_gt_coor = spconv.SparseConvTensor(
            f_center.detach(),
            coordinates_bev.int(),
            sparse_shape,
            self.batch_size
        ).dense()

        bs, _, d, h, w = pts_gt_coor.shape
        pts_gt_coor = pts_gt_coor.reshape(bs, m, -1, h, w)
        batch['gt_coor'] = pts_gt_coor

    def get_ids_keep_mask(self, batch):
        voxels_out, coordinates_out = batch['voxels_out'], batch['coordinates_out']
        coor_down_sample = coordinates_out.int().detach().clone()
        coor_down_sample[:, 1:] //= (self.down_factor * self.grid)
        coor_down_sample[:, 1] = 0

        unique_coor_down_sample, inverse_index = torch.unique(coor_down_sample, return_inverse=True, dim=0)
        batch_indices = unique_coor_down_sample[:, 0].long()
        y_indices = unique_coor_down_sample[:, 2].long()
        x_indices = unique_coor_down_sample[:, 3].long()

        if self.mask_with_sam:
            with torch.no_grad():
                self.model_points.eval()
                output_bev, _ = self.model_points(voxels_out, coordinates_out)
                self.model_points.train()

            output_bev_3 = self.model_adapt_SAM(output_bev)
            output_bev_3 = F.normalize(output_bev_3, p=2, dim=1)
            device = output_bev.device

            if self.prompt == "box":
                # box prompt
                box_key_out = batch['box_key_out']
                box_value_out = batch['box_value_out']
                box_label_out = (torch.ones(box_value_out.shape[0], box_value_out.shape[1], 2) * torch.tensor([2, 3])).int()

                output_mask = self.sam_util(output_bev_3, box_value_out, box_label_out)
                bev_mask_label, _ = torch.where(output_mask, box_key_out.unsqueeze(2).unsqueeze(3), torch.tensor(-1).to(device)).max(dim=1)
            else:
                # point prompt
                point_key_out = batch['point_key_out']
                point_value_out = batch['point_value_out']
                point_value_out = point_value_out.unsqueeze(2)
                point_label_out = (torch.ones(point_value_out.shape[0], point_value_out.shape[1], 1)).int()

                output_mask = self.sam_util(output_bev_3, point_value_out, point_label_out)
                output_mask = output_mask * (torch.sum(output_mask, dim=(2, 3)) <= 100).unsqueeze(-1).unsqueeze(-1)

                bev_mask_label, _ = torch.where(output_mask, point_key_out.unsqueeze(2).unsqueeze(3), torch.tensor(-1).to(device)).max(dim=1)
            
            if self.exc_road:
                bev_mask = (bev_mask_label != -1) & (bev_mask_label != 2)
            else:
                bev_mask = (bev_mask_label!=-1)

        bev_mask_union = batch['bev_mask_union']
        if self.mask_with_sam:
            bev_mask_union = bev_mask_union & (~bev_mask)
        bev_mask_union_len = torch.tensor(len(bev_mask_union[0]) - 1, dtype=torch.long)
        mask_values = bev_mask_union[batch_indices, bev_mask_union_len - y_indices, x_indices]
        keep_indices = ~mask_values
        ids_keep = torch.gather(keep_indices, 0, inverse_index).bool()
        ids_mask = ~ids_keep

        return ids_keep, ids_mask

    # @profile(stream=open('/opt/data/private/memory/train_step.log', 'w+'))
    def training_step(self, batch, batch_idx):
        self.model_points.train()
        if self.model_decoder is not None:
            self.model_decoder.train()
        if self.model_decoder_coor is not None:
            self.model_decoder_coor.train()
        if self.model_decoder_sem is not None:
            self.model_decoder_sem.train()
        if self.model_SAM is not None:
            self.model_SAM.eval()
            self.model_adapt_SAM.train()
        if self.model_teacher is not None:
            self.model_teacher.eval()

        self.batch_size = batch['batch_size']
        ids_keep, ids_mask = self.get_ids_keep_mask(batch)

        # mask
        voxels_out, coordinates_out = batch['voxels_out'], batch['coordinates_out']
        coords_mask = coordinates_out[ids_mask, :]

        # gt
        self.get_chamfer_and_gt_coor(batch)

        # input
        voxel_partial, coords_partial = voxels_out[ids_keep, :], coordinates_out[ids_keep, :]
        average_features = self.model_mask_token.repeat(coords_mask.size(0), 1)
        voxel_partial = torch.cat([voxel_partial, average_features], dim=0)
        coords_partial = torch.cat([coords_partial, coords_mask], dim=0).to(torch.int32)
        batch['voxel_partial'] = voxel_partial
        batch['coords_partial'] = coords_partial

        output_fmap = self.model_points(voxel_partial, coords_partial)
        if isinstance(output_fmap, tuple):
            output_fmap, occupancy_out = output_fmap
        device = output_fmap.device
        batch['output_fmap'] = output_fmap

        if self.model_decoder is not None:
            output_fmap = self.model_decoder(output_fmap)

        bev_mask_union = batch['bev_mask_union']
        batch['gt_mask'] = torch.logical_and(bev_mask_union, occupancy_out)

        output_bev = output_fmap

        # each loss is applied independtly on each GPU
        losses = torch.tensor(0.0).to(device)
        count = 0
        for loss in self.model_losses:
            if loss == "loss_sem":
                losses += self.loss_sem(batch, device, output_bev, occupancy_out)
            elif loss == "loss_coor":
                losses += self.loss_coor(batch, device, output_bev, occupancy_out)
            elif loss == "loss_sem_mse":
                losses += self.loss_sem_mse(batch, device, output_bev, occupancy_out)
            elif loss == "loss_sem_mse_sam_box_switch":
                losses += self.loss_sem_mse_sam_box_switch(batch, device, output_bev, occupancy_out)
            elif loss == "loss_sem_mse_sam_point_switch":
                losses += self.loss_sem_mse_sam_point_switch(batch, device, output_bev, occupancy_out)
            elif loss == "loss_distill":
                losses += self.loss_distill(batch, device, output_bev, occupancy_out)
            elif loss == "loss_distill_l1":
                losses += self.loss_distill(batch, device, output_bev, occupancy_out, loss="L1")
            else:
                raise Exception("Unknown loss")
            count += 1

        if count > 0:
            loss = losses / count
        else:
            raise Exception("Loss is null")
        
        torch.cuda.empty_cache()
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size
        )

        if not self.saved:
            if self.epoch == 20:
                self.save()
                self.saved = True
        return loss
    
    def loss_distill(self, batch, device, output_bev, occupancy_out, loss="L2"):
        if self.model_teacher is None:
            raise Exception("Model_teacher is None")
        
        voxel_partial, coords_partial = batch['voxel_partial'], batch['coords_partial']
        bev_teacher, _ = self.model_teacher(voxel_partial, coords_partial)
        
        output_fmap = batch['output_fmap']

        if loss == "L2":
            loss = self.mse_loss(bev_teacher.detach(), output_fmap)
        else:
            loss = self.L1_loss(bev_teacher.detach(), output_fmap)

        return loss
    
    def sam_util(self, image_tensor, pts_sampled, pts_labels):
        with torch.no_grad():
            device = next(self.model_SAM.parameters()).device
            pts_sampled = pts_sampled.to(device)
            pts_labels = pts_labels.to(device)
            predicted_logits, predicted_iou = self.model_SAM(
                image_tensor,
                pts_sampled,
                pts_labels,
            ) 
            sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
            predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
            predicted_logits = torch.take_along_dim(
                predicted_logits, sorted_ids[..., None, None], dim=2
            )
            return torch.ge(predicted_logits[:, :, 0, :, :], 0)
    
    def loss_sem_mse_sam_box_switch(self, batch, device, output_bev, occupancy_out):
        """
        batch:
            voxels_out
            coordinates_out
            R_out
            T_out

            voxels_bev
            coordinates_bev
            num_points_bev

            bev_map_out
            bev_mask_sem
            bev_mask_union

            gt_mask
            gt_coor
            chamfer_mask
        """
        if self.epoch < self.num_epochs // 2:
            return self.loss_sem_mse(batch, device, output_bev, occupancy_out)
        bev_map_out = batch["bev_map_out"]
        gt_mask = batch["gt_mask"].detach()

        # box prompt
        box_key_out = batch['box_key_out']
        box_value_out = batch['box_value_out']
        box_label_out = (torch.ones(box_value_out.shape[0], box_value_out.shape[1], 2) * torch.tensor([2, 3])).int()

        output_bev_3 = self.model_adapt_SAM(output_bev)
        output_bev_3 = F.normalize(output_bev_3, p=2, dim=1)

        output_mask = self.sam_util(output_bev_3, box_value_out, box_label_out)
        bev_mask_label, _ = torch.where(output_mask, box_key_out.unsqueeze(2).unsqueeze(3), torch.tensor(-1).to(device)).max(dim=1)
        bev_mask = (bev_mask_label!=-1)

        correct_bev_map_out = bev_map_out * (~ bev_mask) + bev_mask_label * bev_mask
        occupancy_bev = (correct_bev_map_out!=-1)

        mask = torch.logical_and(occupancy_bev, gt_mask)

        output_bev = F.normalize(output_bev, p=2, dim=1)
        output_bev_pred = F.conv2d(output_bev, self.text_embeddings[:, :, None, None])

        s = mask.sum().item()
        if s < 4096:
            k = output_bev_pred.permute(0, 2, 3, 1)[mask]
            q = correct_bev_map_out[mask]
        else:
            c = np.random.choice(s, 4096, replace=False)
            k = output_bev_pred.permute(0, 2, 3, 1)[mask][c]
            q = correct_bev_map_out[mask][c]
            
        k = F.normalize(k, p=2, dim=1)
        q = F.one_hot(q, num_classes = self.text_categories).float()

        loss = self.mse_loss(k, q)

        return loss
    
    def loss_sem_mse_sam_point_switch(self, batch, device, output_bev, occupancy_out):
        """
        batch:
            voxels_out
            coordinates_out
            R_out
            T_out

            voxels_bev
            coordinates_bev
            num_points_bev

            bev_map_out
            bev_mask_sem
            bev_mask_union

            gt_mask
            gt_coor
            chamfer_mask
        """
        if self.epoch < self.num_epochs // 2:
            return self.loss_sem_mse(batch, device, output_bev, occupancy_out)
        bev_map_out = batch["bev_map_out"]
        gt_mask = batch["gt_mask"].detach()

        # point prompt
        point_key_out = batch['point_key_out']
        point_value_out = batch['point_value_out']
        point_value_out = point_value_out.unsqueeze(2)
        point_label_out = (torch.ones(point_value_out.shape[0], point_value_out.shape[1], 1)).int()

        output_bev_3 = self.model_adapt_SAM(output_bev)
        output_bev_3 = F.normalize(output_bev_3, p=2, dim=1)

        output_mask = self.sam_util(output_bev_3, point_value_out, point_label_out)
        output_mask = output_mask * (torch.sum(output_mask, dim=(2, 3)) <= 100).unsqueeze(-1).unsqueeze(-1)

        bev_mask_label, _ = torch.where(output_mask, point_key_out.unsqueeze(2).unsqueeze(3), torch.tensor(-1).to(device)).max(dim=1)
        bev_mask = (bev_mask_label!=-1)

        correct_bev_map_out = bev_map_out * (~ bev_mask) + bev_mask_label * bev_mask
        occupancy_bev = (correct_bev_map_out!=-1)

        mask = torch.logical_and(occupancy_bev, gt_mask)

        output_bev = F.normalize(output_bev, p=2, dim=1)
        output_bev_pred = F.conv2d(output_bev, self.text_embeddings[:, :, None, None])
        
        s = mask.sum().item()
        if s < 4096:
            k = output_bev_pred.permute(0, 2, 3, 1)[mask]
            q = correct_bev_map_out[mask]
        else:
            c = np.random.choice(s, 4096, replace=False)
            k = output_bev_pred.permute(0, 2, 3, 1)[mask][c]
            q = correct_bev_map_out[mask][c]
            
        k = F.normalize(k, p=2, dim=1)
        q = F.one_hot(q, num_classes = self.text_categories).float()

        loss = self.mse_loss(k, q)

        return loss

    def label_smoothing(self, labels, num_classes, smoothing=0.1):
        """
        Apply label smoothing to one-hot encoded labels.
        
        Args:
            labels (Tensor): Tensor of class indices (shape: [N]).
            num_classes (int): Total number of classes.
            smoothing (float): Smoothing factor (0.0 means no smoothing).
        
        Returns:
            Tensor: Smoothed labels (shape: [N, num_classes]).
        """
        assert 0 <= smoothing < 1, "Smoothing factor must be in [0, 1)"
        
        # Convert labels to one-hot
        one_hot = F.one_hot(labels, num_classes=num_classes).float()
        
        # Apply label smoothing
        smooth_labels = one_hot * (1 - smoothing) + (smoothing / num_classes)
        
        return smooth_labels

    def loss_sem_mse(self, batch, device, output_bev, occupancy_out):
        """
        batch:
            voxels_out
            coordinates_out
            R_out
            T_out

            voxels_bev
            coordinates_bev
            num_points_bev

            bev_map_out
            bev_mask_sem
            bev_mask_union

            gt_mask
            gt_coor
            chamfer_mask
        """
        bev_map_out = batch["bev_map_out"]
        gt_mask = batch["gt_mask"].detach()
        mask = bev_map_out != -1
        mask = torch.logical_and(mask, gt_mask)

        if self.model_decoder_sem is not None:
            pred_sem = self.model_decoder_sem(output_bev)

            k = pred_sem.permute(0, 2, 3, 1)[mask]
            q = bev_map_out[mask]

            k = F.normalize(k, p=2, dim=1)
            q = self.label_smoothing(q, self.text_categories, smoothing = self.smooth)

            loss = self.mse_loss(k, q)
        else:
            output_bev = F.normalize(output_bev, p=2, dim=1)
            output_bev_pred = F.conv2d(output_bev, self.text_embeddings[:, :, None, None])
            s = mask.sum().item()
            if s < 4096:
                k = output_bev_pred.permute(0, 2, 3, 1)[mask]
                q = bev_map_out[mask]
            else:
                c = np.random.choice(s, 4096, replace=False)
                k = output_bev_pred.permute(0, 2, 3, 1)[mask][c]
                q = bev_map_out[mask][c]
                
            k = F.normalize(k, p=2, dim=1)
            q = self.label_smoothing(q, self.text_categories, smoothing = self.smooth)

            loss = self.mse_loss(k, q)

        return loss
    
    def loss_sem(self, batch, device, output_bev, occupancy_out):
        """
        batch:
            voxels_out
            coordinates_out
            R_out
            T_out

            voxels_bev
            coordinates_bev
            num_points_bev

            bev_map_out
            bev_mask_sem
            bev_mask_union

            gt_mask
            gt_coor
            chamfer_mask
        """
        bev_map_out = batch["bev_map_out"]
        gt_mask = batch["gt_mask"].detach()
        mask = bev_map_out != -1
        mask = torch.logical_and(mask, gt_mask)

        if self.model_decoder_sem is not None:
            pred_sem = self.model_decoder_sem(output_bev)

            k = pred_sem.permute(0, 2, 3, 1)[mask]
            q = bev_map_out[mask]

            loss = self.CE(k, q)
        else:
            output_bev = F.normalize(output_bev, p=2, dim=1)
            output_bev_pred = F.conv2d(output_bev, self.text_embeddings[:, :, None, None])
            s = mask.sum().item()
            if s < 4096:
                k = output_bev_pred.permute(0, 2, 3, 1)[mask]
                q = bev_map_out[mask]
            else:
                c = np.random.choice(s, 4096, replace=False)
                k = output_bev_pred.permute(0, 2, 3, 1)[mask][c]
                q = bev_map_out[mask][c]

            loss = self.CE(k, q)

        return loss
    
    def loss_coor(self, batch, device, output_bev, occupancy_out):
        if self.model_decoder_coor is None:
            return 0
        pred_coor = self.model_decoder_coor(output_bev)
        gt_coor = batch["gt_coor"].detach()
        gt_mask = batch["gt_mask"].detach()
        chamfer_mask = batch["chamfer_mask"].detach()

        bs, d, _, h, w = gt_coor.shape
        gt_coor = gt_coor.reshape(bs, -1, h, w)
        gt_coor = gt_coor.permute(0, 2, 3, 1)
        pred_coor = pred_coor.permute(0, 2, 3, 1)
        chamfer_mask = chamfer_mask.permute(0, 2, 3, 1)

        gt_mask = gt_mask.squeeze().bool()
        if bs == 1:
            gt_mask = gt_mask.unsqueeze(dim=0)

        pred_coor = pred_coor[gt_mask]
        gt_coor = gt_coor[gt_mask]
        chamfer_mask = chamfer_mask[gt_mask]

        pred_coor = pred_coor.reshape(-1, 3, 20).permute(0, 2, 1)
        gt_coor = gt_coor.reshape(-1, d, 3)

        loss_source, loss_target = self.coor_loss(pred_coor, gt_coor, chamfer_mask)
        loss = loss_source + loss_target

        return loss

    def training_epoch_end(self, outputs):
        self.epoch += 1
        if self.epoch == self.num_epochs:
            self.save()
        return super().training_epoch_end(outputs)

    def validation_step(self, batch, batch_idx):
        self.model_points.eval()
        if self.model_decoder is not None:
            self.model_decoder.eval()
        if self.model_decoder_coor is not None:
            self.model_decoder_coor.eval()
        if self.model_decoder_sem is not None:
            self.model_decoder_sem.eval()
        if self.model_SAM is not None:
            self.model_SAM.eval()
            self.model_adapt_SAM.eval()
        if self.model_teacher is not None:
            self.model_teacher.eval()

        self.batch_size = batch['batch_size']
        ids_keep, ids_mask = self.get_ids_keep_mask(batch)

        # mask
        voxels_out, coordinates_out = batch['voxels_out'], batch['coordinates_out']
        coords_mask = coordinates_out[ids_mask, :]

        # gt
        self.get_chamfer_and_gt_coor(batch)

        # input
        voxel_partial, coords_partial = voxels_out[ids_keep, :], coordinates_out[ids_keep, :]
        average_features = self.model_mask_token.repeat(coords_mask.size(0), 1)
        voxel_partial = torch.cat([voxel_partial, average_features], dim=0)
        coords_partial = torch.cat([coords_partial, coords_mask], dim=0).to(torch.int32)
        batch['voxel_partial'] = voxel_partial
        batch['coords_partial'] = coords_partial

        output_fmap = self.model_points(voxel_partial, coords_partial)
        if isinstance(output_fmap, tuple):
            output_fmap, occupancy_out = output_fmap
        device = output_fmap.device
        batch['output_fmap'] = output_fmap

        if self.model_decoder is not None:
            output_fmap = self.model_decoder(output_fmap)

        bev_mask_union = batch['bev_mask_union']
        batch['gt_mask'] = torch.logical_and(bev_mask_union, occupancy_out)

        output_bev = output_fmap
        
        # each loss is applied independtly on each GPU
        losses = torch.tensor(0.0).to(device)
        count = 0
        for loss in self.model_losses:
            if loss == "loss_sem":
                losses += self.loss_sem(batch, device, output_bev, occupancy_out)
            elif loss == "loss_coor":
                losses += self.loss_coor(batch, device, output_bev, occupancy_out)
            elif loss == "loss_sem_mse":
                losses += self.loss_sem_mse(batch, device, output_bev, occupancy_out)
            elif loss == "loss_sem_mse_sam_box_switch":
                losses += self.loss_sem_mse_sam_box_switch(batch, device, output_bev, occupancy_out)
            elif loss == "loss_sem_mse_sam_point_switch":
                losses += self.loss_sem_mse_sam_point_switch(batch, device, output_bev, occupancy_out)
            elif loss == "loss_distill":
                losses += self.loss_distill(batch, device, output_bev, occupancy_out)
            elif loss == "loss_distill_l1":
                losses += self.loss_distill(batch, device, output_bev, occupancy_out, loss="L1")
            else:
                raise Exception("Unknown loss")
            count += 1

        if count > 0:
            loss = losses / count
        else:
            raise Exception("Loss is null")

        torch.cuda.empty_cache()
        self.log(
            "val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size
        )
        return loss

    @rank_zero_only
    def save(self):
        name = "model_epoch" + str(self.epoch) + ".pt"
        path = os.path.join(self.config.MODEL.SAVE_PATH, name)
        model = {}
        model["model_points"] = self.model_points.state_dict()
        if self.model_decoder is not None:
            model["model_decoder"] = self.model_decoder.state_dict()
        if self.model_decoder_coor is not None:
            model["model_decoder_coor"] = self.model_decoder_coor.state_dict()
        if self.model_decoder_sem is not None:
            model["model_decoder_sem"] = self.model_decoder_sem.state_dict()
        if self.model_adapt_SAM is not None:
            model["model_adapt_SAM"] = self.model_adapt_SAM.state_dict()
        model["epoch"] = self.epoch
        model["config"] = self.config
        torch.save(
            model,
            path,
        )