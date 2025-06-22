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
from pytorch_lightning.utilities import rank_zero_only
# from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pretrain.criterion import NCELoss

# from memory_profiler import profile

class LightningPretrain(pl.LightningModule):
    # @profile(stream=open('/opt/data/private/memory/train_init.log', 'w+'))
    def __init__(self, model_points, model_SAM, model_adapt_SAM, config):
        super().__init__()
        self.model_points = model_points
        self.model_SAM = model_SAM
        self.model_adapt_SAM = model_adapt_SAM
        self.config = config

        self.range = config.DATASET.POINT_CLOUD_RANGE
        self.scale = self.range[3]
        self.input_frames = config.DATASET.INPUT_FRAMES

        self.model_losses = config.MODEL.LOSSES
        self.batch_size = config.OPTIMIZATION.BATCH_SIZE_PER_GPU
        self.num_epochs = config.OPTIMIZATION.NUM_EPOCHS
        self.criterion_pp_bc = NCELoss(0.07)
        self.CE = nn.CrossEntropyLoss()
        
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
        if self.model_adapt_SAM is not None:
            optimizer = optimizer_class(
                list(self.model_points.parameters()) + list(self.model_adapt_SAM.parameters()),
                lr=self.config.OPTIMIZATION.LR,
                weight_decay=self.config.OPTIMIZATION.WEIGHT_DECAY
            )
        else:
            optimizer = optimizer_class(
                self.model_points.parameters(),
                lr=self.config.OPTIMIZATION.LR,
                weight_decay=self.config.OPTIMIZATION.WEIGHT_DECAY
            )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.num_epochs)
        # scheduler = LinearWarmupCosineAnnealingLR(optimizer, self.config.OPTIMIZATION.NUM_EPOCHS // 20, self.config.OPTIMIZATION.NUM_EPOCHS, self.config.OPTIMIZATION.LR, self.config.OPTIMIZATION.LR / 100)
        return [optimizer], [scheduler]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    # @profile(stream=open('/opt/data/private/memory/train_step.log', 'w+'))
    def training_step(self, batch, batch_idx):
        self.model_points.train()
        if self.model_SAM is not None:
            self.model_SAM.eval()
            self.model_adapt_SAM.train()

        input_fmap = self.model_points(batch['voxels_in'], batch['coordinates_in'])
        if isinstance(input_fmap, tuple):
            input_fmap, occupancy_in = input_fmap
        device = input_fmap.device

        output_fmap = self.model_points(batch['voxels_out'], batch['coordinates_out'])
        if isinstance(output_fmap, tuple):
            output_fmap, occupancy_out = output_fmap

        input_bev = input_fmap
        output_bev = output_fmap

        # each loss is applied independtly on each GPU
        losses = torch.tensor(0.0).to(device)
        count = 0
        for loss in self.model_losses:
            if loss == "loss_pp_bevcontrast":
                losses += self.loss_pp_bevcontrast(batch, device, input_bev, output_bev, occupancy_in, occupancy_out)
            elif loss == "loss_tp_bevgrid":
                losses += self.loss_tp_bevgrid(batch, output_bev, occupancy_out)
            elif loss == "loss_tp_bevgrid_switch":
                losses += self.loss_tp_bevgrid_switch(batch, output_bev, occupancy_out)
            elif loss == "loss_tp_bevmask_box":
                losses += self.loss_tp_bevmask_box(batch, device, output_bev, occupancy_out)
            elif loss == "loss_tp_bevmask_box_switch":
                losses += self.loss_tp_bevmask_box_switch(batch, device, output_bev, occupancy_out)
            elif loss == "loss_tp_bevmask_point":
                losses += self.loss_tp_bevmask_point(batch, device, output_bev, occupancy_out)
            elif loss == "loss_tp_bevmask_point_switch":
                losses += self.loss_tp_bevmask_point_switch(batch, device, output_bev, occupancy_out)
            elif loss == "loss_tp_bevmask_box_point":
                losses += self.loss_tp_bevmask_box_point(batch, device, output_bev, occupancy_out)
            elif loss == "loss_tp_bevmask_box_point_switch":
                losses += self.loss_tp_bevmask_box_point_switch(batch, device, output_bev, occupancy_out)
            elif loss == "loss_tp_bevmask_box_point_or":
                losses += self.loss_tp_bevmask_box_point_or(batch, device, output_bev, occupancy_out)
            elif loss == "loss_tp_bevmask_box_point_or_switch":
                losses += self.loss_tp_bevmask_box_point_or_switch(batch, device, output_bev, occupancy_out)
            elif loss == "loss_tp_bevmask_box_point_and":
                losses += self.loss_tp_bevmask_box_point_and(batch, device, output_bev, occupancy_out)
            elif loss == "loss_tp_bevmask_box_point_and_switch":
                losses += self.loss_tp_bevmask_box_point_and_switch(batch, device, output_bev, occupancy_out)
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

    def loss_tp_bevmask_box_point_and(self, batch, device, output_bev, occupancy_out):
        bev_map_out = batch["bev_map_out"]

        # box prompt
        box_key_out = batch['box_key_out']
        box_value_out = batch['box_value_out']
        box_label_out = (torch.ones(box_value_out.shape[0], box_value_out.shape[1], 2) * torch.tensor([2, 3])).int()

        # point prompt
        point_key_out = batch['point_key_out']
        point_value_out = batch['point_value_out']
        point_value_out = point_value_out.unsqueeze(2)
        point_label_out = (torch.ones(point_value_out.shape[0], point_value_out.shape[1], 1)).int()

        assert torch.equal(box_key_out, point_key_out), "box_key and point_key are not equal"

        output_bev_3 = self.model_adapt_SAM(output_bev)
        output_bev_3 = F.normalize(output_bev_3, p=2, dim=1)

        # fusion mask:and
        output_mask_box = self.sam_util(output_bev_3, box_value_out, box_label_out)
        output_mask_point = self.sam_util(output_bev_3, point_value_out, point_label_out)
        output_mask_point = output_mask_point * (torch.sum(output_mask_point, dim=(2, 3)) <= 100).unsqueeze(-1).unsqueeze(-1)
        output_mask = torch.logical_and(output_mask_box, output_mask_point)

        # B x 256 x 256
        bev_mask_label, _ = torch.where(output_mask, box_key_out.unsqueeze(2).unsqueeze(3), torch.tensor(-1).to(device)).max(dim=1)
        bev_mask = (bev_mask_label!=-1)

        # Correct bev_map with bev_mask_label
        correct_bev_map_out = bev_map_out * (~ bev_mask) + bev_mask_label * bev_mask

        # The following is consistent with 'loss_tp_bevgrid'
        occupancy_bev = (correct_bev_map_out!=-1)

        output_bev = F.normalize(output_bev, p=2, dim=1)
        output_bev_pred = F.conv2d(output_bev, self.text_embeddings[:, :, None, None])

        mask = torch.logical_and(occupancy_out, occupancy_bev)

        s = mask.sum().item()
        if s < 4096:
            k = output_bev_pred.permute(0, 2, 3, 1)[mask]
            q = correct_bev_map_out[mask]
        else:
            c = np.random.choice(s, 4096, replace=False)
            k = output_bev_pred.permute(0, 2, 3, 1)[mask][c]
            q = correct_bev_map_out[mask][c]

        loss = self.CE(k, q)

        return loss

    
    def loss_tp_bevmask_box_point_and_switch(self, batch, device, output_bev, occupancy_out):
        bev_map_out = batch["bev_map_out"]

        # box prompt
        box_key_out = batch['box_key_out']
        box_value_out = batch['box_value_out']
        box_label_out = (torch.ones(box_value_out.shape[0], box_value_out.shape[1], 2) * torch.tensor([2, 3])).int()

        # point prompt
        point_key_out = batch['point_key_out']
        point_value_out = batch['point_value_out']
        point_value_out = point_value_out.unsqueeze(2)
        point_label_out = (torch.ones(point_value_out.shape[0], point_value_out.shape[1], 1)).int()

        assert torch.equal(box_key_out, point_key_out), "box_key and point_key are not equal"

        output_bev_3 = self.model_adapt_SAM(output_bev)
        output_bev_3 = F.normalize(output_bev_3, p=2, dim=1)

        # fusion mask:and
        output_mask_box = self.sam_util(output_bev_3, box_value_out, box_label_out)
        output_mask_point = self.sam_util(output_bev_3, point_value_out, point_label_out)
        output_mask_point = output_mask_point * (torch.sum(output_mask_point, dim=(2, 3)) <= 100).unsqueeze(-1).unsqueeze(-1)
        output_mask = torch.logical_and(output_mask_box, output_mask_point)

        # B x 256 x 256
        bev_mask_label, _ = torch.where(output_mask, box_key_out.unsqueeze(2).unsqueeze(3), torch.tensor(-1).to(device)).max(dim=1)
        bev_mask = (bev_mask_label!=-1)

        # Correct bev_map with bev_mask_label
        correct_bev_map_out = bev_map_out * (~ bev_mask) + bev_mask_label * bev_mask

        # The following is consistent with 'loss_tp_bevgrid'
        occupancy_bev = (correct_bev_map_out!=-1)

        output_bev = F.normalize(output_bev, p=2, dim=1)
        output_bev_pred = F.conv2d(output_bev, self.text_embeddings[:, :, None, None])

        mask = torch.logical_and(occupancy_out, occupancy_bev)

        s = mask.sum().item()
        if s < 4096:
            k = output_bev_pred.permute(0, 2, 3, 1)[mask]
            q = correct_bev_map_out[mask]
        else:
            c = np.random.choice(s, 4096, replace=False)
            k = output_bev_pred.permute(0, 2, 3, 1)[mask][c]
            q = correct_bev_map_out[mask][c]

        # switchable training strategy
        if self.epoch >= (self.num_epochs / 2):
            rd = random.randint(1, 10)
            if rd > 5: q = k.argmax(dim=1)

        loss = self.CE(k, q)

        return loss

    def loss_tp_bevmask_box_point_or(self, batch, device, output_bev, occupancy_out):
        bev_map_out = batch["bev_map_out"]

        # box prompt
        box_key_out = batch['box_key_out']
        box_value_out = batch['box_value_out']
        box_label_out = (torch.ones(box_value_out.shape[0], box_value_out.shape[1], 2) * torch.tensor([2, 3])).int()

        # point prompt
        point_key_out = batch['point_key_out']
        point_value_out = batch['point_value_out']
        point_value_out = point_value_out.unsqueeze(2)
        point_label_out = (torch.ones(point_value_out.shape[0], point_value_out.shape[1], 1)).int()

        assert torch.equal(box_key_out, point_key_out), "box_key and point_key are not equal"

        output_bev_3 = self.model_adapt_SAM(output_bev)
        output_bev_3 = F.normalize(output_bev_3, p=2, dim=1)

        # fusion mask:or
        output_mask_box = self.sam_util(output_bev_3, box_value_out, box_label_out)
        output_mask_point = self.sam_util(output_bev_3, point_value_out, point_label_out)
        output_mask_point = output_mask_point * (torch.sum(output_mask_point, dim=(2, 3)) <= 100).unsqueeze(-1).unsqueeze(-1)
        output_mask = torch.logical_or(output_mask_box, output_mask_point)

        # B x 256 x 256
        bev_mask_label, _ = torch.where(output_mask, box_key_out.unsqueeze(2).unsqueeze(3), torch.tensor(-1).to(device)).max(dim=1)
        bev_mask = (bev_mask_label!=-1)

        # Correct bev_map with bev_mask_label
        correct_bev_map_out = bev_map_out * (~ bev_mask) + bev_mask_label * bev_mask

        # The following is consistent with 'loss_tp_bevgrid'
        occupancy_bev = (correct_bev_map_out!=-1)

        output_bev = F.normalize(output_bev, p=2, dim=1)
        output_bev_pred = F.conv2d(output_bev, self.text_embeddings[:, :, None, None])

        mask = torch.logical_and(occupancy_out, occupancy_bev)

        s = mask.sum().item()
        if s < 4096:
            k = output_bev_pred.permute(0, 2, 3, 1)[mask]
            q = correct_bev_map_out[mask]
        else:
            c = np.random.choice(s, 4096, replace=False)
            k = output_bev_pred.permute(0, 2, 3, 1)[mask][c]
            q = correct_bev_map_out[mask][c]

        loss = self.CE(k, q)

        return loss

    def loss_tp_bevmask_box_point_or_switch(self, batch, device, output_bev, occupancy_out):
        bev_map_out = batch["bev_map_out"]

        # box prompt
        box_key_out = batch['box_key_out']
        box_value_out = batch['box_value_out']
        box_label_out = (torch.ones(box_value_out.shape[0], box_value_out.shape[1], 2) * torch.tensor([2, 3])).int()

        # point prompt
        point_key_out = batch['point_key_out']
        point_value_out = batch['point_value_out']
        point_value_out = point_value_out.unsqueeze(2)
        point_label_out = (torch.ones(point_value_out.shape[0], point_value_out.shape[1], 1)).int()

        assert torch.equal(box_key_out, point_key_out), "box_key and point_key are not equal"

        output_bev_3 = self.model_adapt_SAM(output_bev)
        output_bev_3 = F.normalize(output_bev_3, p=2, dim=1)

        # fusion mask:or
        output_mask_box = self.sam_util(output_bev_3, box_value_out, box_label_out)
        output_mask_point = self.sam_util(output_bev_3, point_value_out, point_label_out)
        output_mask_point = output_mask_point * (torch.sum(output_mask_point, dim=(2, 3)) <= 100).unsqueeze(-1).unsqueeze(-1)
        output_mask = torch.logical_or(output_mask_box, output_mask_point)

        # B x 256 x 256
        bev_mask_label, _ = torch.where(output_mask, box_key_out.unsqueeze(2).unsqueeze(3), torch.tensor(-1).to(device)).max(dim=1)
        bev_mask = (bev_mask_label!=-1)

        # Correct bev_map with bev_mask_label
        correct_bev_map_out = bev_map_out * (~ bev_mask) + bev_mask_label * bev_mask

        # The following is consistent with 'loss_tp_bevgrid'
        occupancy_bev = (correct_bev_map_out!=-1)

        output_bev = F.normalize(output_bev, p=2, dim=1)
        output_bev_pred = F.conv2d(output_bev, self.text_embeddings[:, :, None, None])

        mask = torch.logical_and(occupancy_out, occupancy_bev)

        s = mask.sum().item()
        if s < 4096:
            k = output_bev_pred.permute(0, 2, 3, 1)[mask]
            q = correct_bev_map_out[mask]
        else:
            c = np.random.choice(s, 4096, replace=False)
            k = output_bev_pred.permute(0, 2, 3, 1)[mask][c]
            q = correct_bev_map_out[mask][c]

        # switchable training strategy
        if self.epoch >= (self.num_epochs / 2):
            rd = random.randint(1, 10)
            if rd > 5: q = k.argmax(dim=1)

        loss = self.CE(k, q)

        return loss

    def loss_tp_bevmask_box_point(self, batch, device, output_bev, occupancy_out):
        bev_map_out = batch["bev_map_out"]

        # box prompt
        box_key_out = batch['box_key_out']
        box_value_out = batch['box_value_out']
        box_label_out = (torch.ones(box_value_out.shape[0], box_value_out.shape[1], 2) * torch.tensor([2, 3])).int()

        # point prompt
        point_key_out = batch['point_key_out']
        point_value_out = batch['point_value_out']
        point_value_out = point_value_out.unsqueeze(2)
        point_label_out = (torch.ones(point_value_out.shape[0], point_value_out.shape[1], 1)).int()

        assert torch.equal(box_key_out, point_key_out), "box_key and point_key are not equal"

        output_bev_3 = self.model_adapt_SAM(output_bev)
        output_bev_3 = F.normalize(output_bev_3, p=2, dim=1)

        # fusion mask
        output_mask_box = self.sam_util(output_bev_3, box_value_out, box_label_out)
        output_mask_point = self.sam_util(output_bev_3, point_value_out, point_label_out)
        output_mask = torch.logical_and(output_mask_box, output_mask_point)

        # B x 256 x 256
        bev_mask_label, _ = torch.where(output_mask, box_key_out.unsqueeze(2).unsqueeze(3), torch.tensor(-1).to(device)).max(dim=1)
        bev_mask = (bev_mask_label!=-1)

        # Correct bev_map with bev_mask_label
        correct_bev_map_out = bev_map_out * (~ bev_mask) + bev_mask_label * bev_mask

        # The following is consistent with 'loss_tp_bevgrid'
        occupancy_bev = (correct_bev_map_out!=-1)

        output_bev = F.normalize(output_bev, p=2, dim=1)
        output_bev_pred = F.conv2d(output_bev, self.text_embeddings[:, :, None, None])

        mask = torch.logical_and(occupancy_out, occupancy_bev)

        s = mask.sum().item()
        if s < 4096:
            k = output_bev_pred.permute(0, 2, 3, 1)[mask]
            q = correct_bev_map_out[mask]
        else:
            c = np.random.choice(s, 4096, replace=False)
            k = output_bev_pred.permute(0, 2, 3, 1)[mask][c]
            q = correct_bev_map_out[mask][c]

        loss = self.CE(k, q)

        return loss

    def loss_tp_bevmask_box_point_switch(self, batch, device, output_bev, occupancy_out):
        bev_map_out = batch["bev_map_out"]

        # box prompt
        box_key_out = batch['box_key_out']
        box_value_out = batch['box_value_out']
        box_label_out = (torch.ones(box_value_out.shape[0], box_value_out.shape[1], 2) * torch.tensor([2, 3])).int()

        # point prompt
        point_key_out = batch['point_key_out']
        point_value_out = batch['point_value_out']
        point_value_out = point_value_out.unsqueeze(2)
        point_label_out = (torch.ones(point_value_out.shape[0], point_value_out.shape[1], 1)).int()

        assert torch.equal(box_key_out, point_key_out), "box_key and point_key are not equal"

        output_bev_3 = self.model_adapt_SAM(output_bev)
        output_bev_3 = F.normalize(output_bev_3, p=2, dim=1)

        # fusion mask
        output_mask_box = self.sam_util(output_bev_3, box_value_out, box_label_out)
        output_mask_point = self.sam_util(output_bev_3, point_value_out, point_label_out)
        output_mask = torch.logical_and(output_mask_box, output_mask_point)

        # B x 256 x 256
        bev_mask_label, _ = torch.where(output_mask, box_key_out.unsqueeze(2).unsqueeze(3), torch.tensor(-1).to(device)).max(dim=1)
        bev_mask = (bev_mask_label!=-1)

        # Correct bev_map with bev_mask_label
        correct_bev_map_out = bev_map_out * (~ bev_mask) + bev_mask_label * bev_mask

        # The following is consistent with 'loss_tp_bevgrid'
        occupancy_bev = (correct_bev_map_out!=-1)

        output_bev = F.normalize(output_bev, p=2, dim=1)
        output_bev_pred = F.conv2d(output_bev, self.text_embeddings[:, :, None, None])

        mask = torch.logical_and(occupancy_out, occupancy_bev)

        s = mask.sum().item()
        if s < 4096:
            k = output_bev_pred.permute(0, 2, 3, 1)[mask]
            q = correct_bev_map_out[mask]
        else:
            c = np.random.choice(s, 4096, replace=False)
            k = output_bev_pred.permute(0, 2, 3, 1)[mask][c]
            q = correct_bev_map_out[mask][c]

        # switchable training strategy
        if self.epoch >= (self.num_epochs / 2):
            rd = random.randint(1, 10)
            if rd > 5: q = k.argmax(dim=1)

        loss = self.CE(k, q)

        return loss

    def loss_tp_bevmask_point(self, batch, device, output_bev, occupancy_out):
        bev_map_out = batch["bev_map_out"]
        # B x 5
        point_key_out = batch['point_key_out']
        # B x 5 x 2
        point_value_out = batch['point_value_out']
        # B x 5 x 1 x 2
        point_value_out = point_value_out.unsqueeze(2)
        # B x 5 x 1
        point_label_out = (torch.ones(point_value_out.shape[0], point_value_out.shape[1], 1)).int()

        output_bev_3 = self.model_adapt_SAM(output_bev)
        output_bev_3 = F.normalize(output_bev_3, p=2, dim=1)

        # B x 5 x 256 x 256
        output_mask = self.sam_util(output_bev_3, point_value_out, point_label_out)
        # filter
        output_mask = output_mask * (torch.sum(output_mask, dim=(2, 3)) <= 100).unsqueeze(-1).unsqueeze(-1)

        # B x 256 x 256
        bev_mask_label, _ = torch.where(output_mask, point_key_out.unsqueeze(2).unsqueeze(3), torch.tensor(-1).to(device)).max(dim=1)
        bev_mask = (bev_mask_label!=-1)

        # Correct bev_map with bev_mask_label
        correct_bev_map_out = bev_map_out * (~ bev_mask) + bev_mask_label * bev_mask

        # The following is consistent with 'loss_tp_bevgrid'
        occupancy_bev = (correct_bev_map_out!=-1)

        output_bev = F.normalize(output_bev, p=2, dim=1)
        output_bev_pred = F.conv2d(output_bev, self.text_embeddings[:, :, None, None])

        mask = torch.logical_and(occupancy_out, occupancy_bev)

        s = mask.sum().item()
        if s < 4096:
            k = output_bev_pred.permute(0, 2, 3, 1)[mask]
            q = correct_bev_map_out[mask]
        else:
            c = np.random.choice(s, 4096, replace=False)
            k = output_bev_pred.permute(0, 2, 3, 1)[mask][c]
            q = correct_bev_map_out[mask][c]

        loss = self.CE(k, q)

        return loss

    def loss_tp_bevmask_point_switch(self, batch, device, output_bev, occupancy_out):
        bev_map_out = batch["bev_map_out"]
        # B x 5
        point_key_out = batch['point_key_out']
        # B x 5 x 2
        point_value_out = batch['point_value_out']
        # B x 5 x 1 x 2
        point_value_out = point_value_out.unsqueeze(2)
        # B x 5 x 1
        point_label_out = (torch.ones(point_value_out.shape[0], point_value_out.shape[1], 1)).int()

        output_bev_3 = self.model_adapt_SAM(output_bev)
        output_bev_3 = F.normalize(output_bev_3, p=2, dim=1)

        # B x 5 x 256 x 256
        output_mask = self.sam_util(output_bev_3, point_value_out, point_label_out)
        # filter
        output_mask = output_mask * (torch.sum(output_mask, dim=(2, 3)) <= 100).unsqueeze(-1).unsqueeze(-1)

        # B x 256 x 256
        bev_mask_label, _ = torch.where(output_mask, point_key_out.unsqueeze(2).unsqueeze(3), torch.tensor(-1).to(device)).max(dim=1)
        bev_mask = (bev_mask_label!=-1)

        # Correct bev_map with bev_mask_label
        correct_bev_map_out = bev_map_out * (~ bev_mask) + bev_mask_label * bev_mask

        # The following is consistent with 'loss_tp_bevgrid'
        occupancy_bev = (correct_bev_map_out!=-1)

        output_bev = F.normalize(output_bev, p=2, dim=1)
        output_bev_pred = F.conv2d(output_bev, self.text_embeddings[:, :, None, None])

        mask = torch.logical_and(occupancy_out, occupancy_bev)

        s = mask.sum().item()
        if s < 4096:
            k = output_bev_pred.permute(0, 2, 3, 1)[mask]
            q = correct_bev_map_out[mask]
        else:
            c = np.random.choice(s, 4096, replace=False)
            k = output_bev_pred.permute(0, 2, 3, 1)[mask][c]
            q = correct_bev_map_out[mask][c]

        # switchable training strategy
        if self.epoch >= (self.num_epochs / 2):
            rd = random.randint(1, 10)
            if rd > 5: q = k.argmax(dim=1)

        loss = self.CE(k, q)

        return loss
    
    def loss_tp_bevmask_box(self, batch, device, output_bev, occupancy_out):
        bev_map_out = batch["bev_map_out"]
        # B x 5 
        box_key_out = batch['box_key_out']
        # B x 5 x 2 x 2
        box_value_out = batch['box_value_out']
        # B x 5 x 2
        box_label_out = (torch.ones(box_value_out.shape[0], box_value_out.shape[1], 2) * torch.tensor([2, 3])).int()

        output_bev_3 = self.model_adapt_SAM(output_bev)
        output_bev_3 = F.normalize(output_bev_3, p=2, dim=1)

        # B x 5 x 256 x 256
        output_mask = self.sam_util(output_bev_3, box_value_out, box_label_out)

        # B x 256 x 256
        bev_mask_label, _ = torch.where(output_mask, box_key_out.unsqueeze(2).unsqueeze(3), torch.tensor(-1).to(device)).max(dim=1)
        bev_mask = (bev_mask_label!=-1)

        # Correct bev_map with bev_mask_label
        correct_bev_map_out = bev_map_out * (~ bev_mask) + bev_mask_label * bev_mask

        # The following is consistent with 'loss_tp_bevgrid'
        occupancy_bev = (correct_bev_map_out!=-1)

        output_bev = F.normalize(output_bev, p=2, dim=1)
        output_bev_pred = F.conv2d(output_bev, self.text_embeddings[:, :, None, None])

        mask = torch.logical_and(occupancy_out, occupancy_bev)

        s = mask.sum().item()
        if s < 4096:
            k = output_bev_pred.permute(0, 2, 3, 1)[mask]
            q = correct_bev_map_out[mask]
        else:
            c = np.random.choice(s, 4096, replace=False)
            k = output_bev_pred.permute(0, 2, 3, 1)[mask][c]
            q = correct_bev_map_out[mask][c]

        loss = self.CE(k, q)

        return loss

    def loss_tp_bevmask_box_switch(self, batch, device, output_bev, occupancy_out):
        bev_map_out = batch["bev_map_out"]
        # B x 5 
        box_key_out = batch['box_key_out']
        # B x 5 x 2 x 2
        box_value_out = batch['box_value_out']
        # B x 5 x 2
        box_label_out = (torch.ones(box_value_out.shape[0], box_value_out.shape[1], 2) * torch.tensor([2, 3])).int()

        output_bev_3 = self.model_adapt_SAM(output_bev)
        output_bev_3 = F.normalize(output_bev_3, p=2, dim=1)

        # B x 5 x 256 x 256
        output_mask = self.sam_util(output_bev_3, box_value_out, box_label_out)

        # B x 256 x 256
        bev_mask_label, _ = torch.where(output_mask, box_key_out.unsqueeze(2).unsqueeze(3), torch.tensor(-1).to(device)).max(dim=1)
        bev_mask = (bev_mask_label!=-1)

        # Correct bev_map with bev_mask_label
        correct_bev_map_out = bev_map_out * (~ bev_mask) + bev_mask_label * bev_mask

        # The following is consistent with 'loss_tp_bevgrid'
        occupancy_bev = (correct_bev_map_out!=-1)

        output_bev = F.normalize(output_bev, p=2, dim=1)
        output_bev_pred = F.conv2d(output_bev, self.text_embeddings[:, :, None, None])

        mask = torch.logical_and(occupancy_out, occupancy_bev)

        s = mask.sum().item()
        if s < 4096:
            k = output_bev_pred.permute(0, 2, 3, 1)[mask]
            q = correct_bev_map_out[mask]
        else:
            c = np.random.choice(s, 4096, replace=False)
            k = output_bev_pred.permute(0, 2, 3, 1)[mask][c]
            q = correct_bev_map_out[mask][c]

        # switchable training strategy
        if self.epoch >= (self.num_epochs / 2):
            rd = random.randint(1, 10)
            if rd > 5: q = k.argmax(dim=1)

        loss = self.CE(k, q)

        return loss

    def loss_tp_bevgrid(self, batch, output_bev, occupancy_out):
        bev_map_out = batch["bev_map_out"]
        occupancy_bev = (bev_map_out!=-1)

        output_bev = F.normalize(output_bev, p=2, dim=1)
        output_bev_pred = F.conv2d(output_bev, self.text_embeddings[:, :, None, None])

        mask = torch.logical_and(occupancy_out, occupancy_bev)

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


    def loss_tp_bevgrid_switch(self, batch, output_bev, occupancy_out):
        bev_map_out = batch["bev_map_out"]
        occupancy_bev = (bev_map_out!=-1)

        output_bev = F.normalize(output_bev, p=2, dim=1)
        output_bev_pred = F.conv2d(output_bev, self.text_embeddings[:, :, None, None])

        mask = torch.logical_and(occupancy_out, occupancy_bev)

        s = mask.sum().item()
        if s < 4096:
            k = output_bev_pred.permute(0, 2, 3, 1)[mask]
            q = bev_map_out[mask]
        else:
            c = np.random.choice(s, 4096, replace=False)
            k = output_bev_pred.permute(0, 2, 3, 1)[mask][c]
            q = bev_map_out[mask][c]

        # switchable training strategy
        if self.epoch >= (self.num_epochs / 2):
            rd = random.randint(1, 10)
            if rd > 5: q = k.argmax(dim=1)

        loss = self.CE(k, q)

        return loss
    
    # @profile(stream=open('/opt/data/private/memory/train_bc.log', 'w+'))
    def loss_pp_bevcontrast(self, batch, device, input_bev, output_bev, occupancy_in, occupancy_out):
        batch = copy.deepcopy(batch)
        # recover R and T on the BEV plane
        R = (batch["R_out"].transpose(1, 2) @ batch["R_in"]).to(torch.float32)
        T = ((batch["T_in"] - batch["T_out"]).unsqueeze(1) @ batch["R_out"].to(torch.float32))
        Rim = R.transpose(1, 2)
        Tim = -T @ R
        Rim = Rim[:, :2, :2]
        Tim = Tim[:, 0, :2] / self.scale
        P = torch.cat([Rim, Tim.unsqueeze(2)], axis=2)
        grid = F.affine_grid(P.to(device, non_blocking=True), output_bev.shape, align_corners=False)

        pred_bev = F.grid_sample(input_bev[0::self.input_frames], grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        occupancy_pred = F.grid_sample(occupancy_in[0::self.input_frames].unsqueeze(1).to(torch.float32), grid, mode='bilinear', padding_mode='zeros', align_corners=False).squeeze(1).bool()
        mask = torch.logical_and(occupancy_out, occupancy_pred)

        s = mask.sum().item()
        if s < 4096:
            k = F.normalize(pred_bev.permute(0, 2, 3, 1)[mask], p=2, dim=1)
            q = F.normalize(output_bev.permute(0, 2, 3, 1)[mask], p=2, dim=1)
        else:
            c = np.random.choice(s, 4096, replace=False)
            k = F.normalize(pred_bev.permute(0, 2, 3, 1)[mask][c], p=2, dim=1)
            q = F.normalize(output_bev.permute(0, 2, 3, 1)[mask][c], p=2, dim=1)
        loss = self.criterion_pp_bc(k, q)
        return loss


    def training_epoch_end(self, outputs):
        self.epoch += 1
        if self.epoch == self.num_epochs:
            self.save()
        return super().training_epoch_end(outputs)

    def validation_step(self, batch, batch_idx):
        self.model_points.eval()
        if self.model_SAM is not None:
            self.model_SAM.eval()
            self.model_adapt_SAM.eval()

        input_fmap = self.model_points(batch['voxels_in'], batch['coordinates_in'])
        if isinstance(input_fmap, tuple):
            input_fmap, occupancy_in = input_fmap
        device = input_fmap.device

        output_fmap = self.model_points(batch['voxels_out'], batch['coordinates_out'])
        if isinstance(output_fmap, tuple):
            output_fmap, occupancy_out = output_fmap

        input_bev = input_fmap
        output_bev = output_fmap

        # each loss is applied independtly on each GPU
        losses = torch.tensor(0.0).to(device)
        count = 0
        for loss in self.model_losses:
            if loss == "loss_pp_bevcontrast":
                losses += self.loss_pp_bevcontrast(batch, device, input_bev, output_bev, occupancy_in, occupancy_out)
            elif loss == "loss_tp_bevgrid":
                losses += self.loss_tp_bevgrid(batch, output_bev, occupancy_out)
            elif loss == "loss_tp_bevgrid_switch":
                losses += self.loss_tp_bevgrid_switch(batch, output_bev, occupancy_out)
            elif loss == "loss_tp_bevmask_box":
                losses += self.loss_tp_bevmask_box(batch, device, output_bev, occupancy_out)
            elif loss == "loss_tp_bevmask_box_switch":
                losses += self.loss_tp_bevmask_box_switch(batch, device, output_bev, occupancy_out)
            elif loss == "loss_tp_bevmask_point":
                losses += self.loss_tp_bevmask_point(batch, device, output_bev, occupancy_out)
            elif loss == "loss_tp_bevmask_point_switch":
                losses += self.loss_tp_bevmask_point_switch(batch, device, output_bev, occupancy_out)
            elif loss == "loss_tp_bevmask_box_point":
                losses += self.loss_tp_bevmask_box_point(batch, device, output_bev, occupancy_out)
            elif loss == "loss_tp_bevmask_box_point_switch":
                losses += self.loss_tp_bevmask_box_point_switch(batch, device, output_bev, occupancy_out)
            elif loss == "loss_tp_bevmask_box_point_or":
                losses += self.loss_tp_bevmask_box_point_or(batch, device, output_bev, occupancy_out)
            elif loss == "loss_tp_bevmask_box_point_or_switch":
                losses += self.loss_tp_bevmask_box_point_or_switch(batch, device, output_bev, occupancy_out)
            elif loss == "loss_tp_bevmask_box_point_and":
                losses += self.loss_tp_bevmask_box_point_and(batch, device, output_bev, occupancy_out)
            elif loss == "loss_tp_bevmask_box_point_and_switch":
                losses += self.loss_tp_bevmask_box_point_and_switch(batch, device, output_bev, occupancy_out)
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
        if self.model_adapt_SAM is not None:
            torch.save(
                {
                    "model_points": self.model_points.state_dict(),
                    "model_adapt_SAM": self.model_adapt_SAM.state_dict(),
                    "epoch": self.epoch,
                    "config": self.config,
                },
                path,
            )
        else:
            torch.save(
                {
                    "model_points": self.model_points.state_dict(),
                    "epoch": self.epoch,
                    "config": self.config,
                },
                path,
            )