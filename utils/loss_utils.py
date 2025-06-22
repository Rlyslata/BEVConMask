# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn as nn
from torch.nn.functional import l1_loss, mse_loss, smooth_l1_loss

def mask_chamfer_distance(src,
                     dst,
                     mask,
                     src_weight=1.0,
                     dst_weight=1.0,
                     criterion_mode='l2',
                     reduction='mean',
                     dynamic_weight=False):
    """Calculate Chamfer Distance of two sets.

    Args:
        src (torch.Tensor): Source set with shape [B, N, C] to
            calculate Chamfer Distance.
        dst (torch.Tensor): Destination set with shape [B, M, C] to
            calculate Chamfer Distance.
        src_weight (torch.Tensor or float): Weight of source loss.
        dst_weight (torch.Tensor or float): Weight of destination loss.
        criterion_mode (str): Criterion mode to calculate distance.
            The valid modes are smooth_l1, l1 or l2.
        reduction (str): Method to reduce losses.
            The valid reduction method are 'none', 'sum' or 'mean'.

    Returns:
        tuple: Source and Destination loss with the corresponding indices.

            - loss_src (torch.Tensor): The min distance \
                from source to destination.
            - loss_dst (torch.Tensor): The min distance \
                from destination to source.
            - indices1 (torch.Tensor): Index the min distance point \
                for each point in source to destination.
            - indices2 (torch.Tensor): Index the min distance point \
                for each point in destination to source.
    """

    if criterion_mode == 'smooth_l1':
        criterion = smooth_l1_loss
    elif criterion_mode == 'l1':
        criterion = l1_loss
    elif criterion_mode == 'l2':
        criterion = mse_loss
    else:
        raise NotImplementedError
    
    assert src.size(0) == dst.size(0)
    assert mask.size()[:2] == dst.size()[:2]

    src_expand = src.unsqueeze(2).repeat(1, 1, dst.shape[1], 1)
    dst_expand = dst.unsqueeze(1).repeat(1, src.shape[1], 1, 1)
    # print(src.size(), dst.size())
    # src_expand = src.unsqueeze(2).expand(-1, -1, dst.shape[1], -1)
    # dst_expand = dst.unsqueeze(1).expand(-1, src.shape[1], -1, -1)

    distance = criterion(src_expand, dst_expand, reduction='none').sum(-1)
    mask_expand = mask.unsqueeze(dim=1).expand_as(distance)

    max_dist = distance.max().detach()
    new_distance = distance - mask_expand*max_dist

    src2dst_distance, indices1 = torch.min(new_distance, dim=2)  # (B,N)

    src2dst_distance = src2dst_distance + max_dist
    dst2src_distance, indices2 = torch.min(distance, dim=1)  # (B,M)

    loss_src = (src2dst_distance * src_weight)
    loss_dst = (dst2src_distance * dst_weight)

    if dynamic_weight:
        dy_weight = mask.sum(dim=1, keepdim=True)
        dy_weight = dy_weight/mask.shape[1]
        # print(dy_weight.max(), dy_weight.min())
        loss_src = loss_src * dy_weight

    if reduction == 'sum':
        loss_src = torch.sum(loss_src)
        loss_dst = torch.sum(loss_dst * mask)
    elif reduction == 'mean':
        loss_src = torch.mean(loss_src)
        loss_dst = (loss_dst*mask).sum()/mask.sum()
    elif reduction == 'none':
        pass
    else:
        raise NotImplementedError

    return loss_src, loss_dst, indices1, indices2


# @LOSSES.register_module()
class MaskChamferDistance(nn.Module):
    """Calculate Chamfer Distance of two sets.

    Args:
        mode (str): Criterion mode to calculate distance.
            The valid modes are smooth_l1, l1 or l2.
        reduction (str): Method to reduce losses.
            The valid reduction method are none, sum or mean.
        loss_src_weight (float): Weight of loss_source.
        loss_dst_weight (float): Weight of loss_target.
    """

    def __init__(self,
                 mode='l2',
                 reduction='mean',
                 loss_src_weight=1.0,
                 loss_dst_weight=1.0,
                 dynamic_weight=False,):
        super(MaskChamferDistance, self).__init__()

        assert mode in ['smooth_l1', 'l1', 'l2']
        assert reduction in ['none', 'sum', 'mean']
        self.mode = mode
        self.reduction = reduction
        self.loss_src_weight = loss_src_weight
        self.loss_dst_weight = loss_dst_weight

        self.dynamic_weight = dynamic_weight

    def forward(self,
                source,
                target,
                mask,
                src_weight=1.0,
                dst_weight=1.0,
                reduction_override=None,
                return_indices=False,
                **kwargs):
        """Forward function of loss calculation.

        Args:
            source (torch.Tensor): Source set with shape [B, N, C] to
                calculate Chamfer Distance.
            target (torch.Tensor): Destination set with shape [B, M, C] to
                calculate Chamfer Distance.
            src_weight (torch.Tensor | float, optional):
                Weight of source loss. Defaults to 1.0.
            dst_weight (torch.Tensor | float, optional):
                Weight of destination loss. Defaults to 1.0.
            reduction_override (str, optional): Method to reduce losses.
                The valid reduction method are 'none', 'sum' or 'mean'.
                Defaults to None.
            return_indices (bool, optional): Whether to return indices.
                Defaults to False.

        Returns:
            tuple[torch.Tensor]: If ``return_indices=True``, return losses of \
                source and target with their corresponding indices in the \
                order of ``(loss_source, loss_target, indices1, indices2)``. \
                If ``return_indices=False``, return \
                ``(loss_source, loss_target)``.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss_source, loss_target, indices1, indices2 = mask_chamfer_distance(
            source, target, mask, src_weight, dst_weight, self.mode, reduction, self.dynamic_weight)

        loss_source *= self.loss_src_weight
        loss_target *= self.loss_dst_weight

        if return_indices:
            return loss_source, loss_target, indices1, indices2
        else:
            return loss_source, loss_target


def chamfer_distance(src,
                     dst,
                     src_weight=1.0,
                     dst_weight=1.0,
                     criterion_mode='l2',
                     reduction='mean'):
    """Calculate Chamfer Distance of two sets.

    Args:
        src (torch.Tensor): Source set with shape [B, N, C] to
            calculate Chamfer Distance.
        dst (torch.Tensor): Destination set with shape [B, M, C] to
            calculate Chamfer Distance.
        src_weight (torch.Tensor or float): Weight of source loss.
        dst_weight (torch.Tensor or float): Weight of destination loss.
        criterion_mode (str): Criterion mode to calculate distance.
            The valid modes are smooth_l1, l1 or l2.
        reduction (str): Method to reduce losses.
            The valid reduction method are 'none', 'sum' or 'mean'.

    Returns:
        tuple: Source and Destination loss with the corresponding indices.

            - loss_src (torch.Tensor): The min distance \
                from source to destination.
            - loss_dst (torch.Tensor): The min distance \
                from destination to source.
            - indices1 (torch.Tensor): Index the min distance point \
                for each point in source to destination.
            - indices2 (torch.Tensor): Index the min distance point \
                for each point in destination to source.
    """

    if criterion_mode == 'smooth_l1':
        criterion = smooth_l1_loss
    elif criterion_mode == 'l1':
        criterion = l1_loss
    elif criterion_mode == 'l2':
        criterion = mse_loss
    else:
        raise NotImplementedError

    src_expand = src.unsqueeze(2).repeat(1, 1, dst.shape[1], 1)
    dst_expand = dst.unsqueeze(1).repeat(1, src.shape[1], 1, 1)
    # print(src.size(), dst.size())
    # src_expand = src.unsqueeze(2).expand(-1, -1, dst.shape[1], -1)
    # dst_expand = dst.unsqueeze(1).expand(-1, src.shape[1], -1, -1)

    distance = criterion(src_expand, dst_expand, reduction='none').sum(-1)
    src2dst_distance, indices1 = torch.min(distance, dim=2)  # (B,N)
    dst2src_distance, indices2 = torch.min(distance, dim=1)  # (B,M)

    loss_src = (src2dst_distance * src_weight)
    loss_dst = (dst2src_distance * dst_weight)

    if reduction == 'sum':
        loss_src = torch.sum(loss_src)
        loss_dst = torch.sum(loss_dst)
    elif reduction == 'mean':
        loss_src = torch.mean(loss_src)
        loss_dst = torch.mean(loss_dst)
    elif reduction == 'none':
        pass
    else:
        raise NotImplementedError

    return loss_src, loss_dst, indices1, indices2

