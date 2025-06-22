from model import (
    SECOND,
)
from efficient_sam.build_efficient_sam import build_efficient_sam_vitt

import os
import torch
import torch.nn as nn

def make_model(config, logger=None):
    """
    Build points, image and SAM models according to what is in the config
    """
    if config.MODEL.POINT_ENCODER:
        if config.POINT_ENCODER.NAME.lower() == "second":
            model_points = SECOND(config.POINT_ENCODER.IN_CHANNELS, config.POINT_ENCODER.OUT_CHANNELS, config=config)  
        else:
            raise Exception(f"Points model not found: {config.POINT_ENCODER.NAME}")
    else:
        raise Exception(f"Points model is empty")
    
    # needed by CMBEV
    if config.MODEL.DECODER:
        if config.DECODER.NAME.lower() == "conv":
            if config.DECODER.SIZE == 1:
                # 1*1 conv
                model_decoder = nn.Sequential(
                    nn.Conv2d(in_channels=config.POINT_ENCODER.FEATURE_DIMENSION, out_channels=config.POINT_ENCODER.FEATURE_DIMENSION, kernel_size=1),
                    nn.BatchNorm2d(config.POINT_ENCODER.FEATURE_DIMENSION),
                    nn.ReLU()
                )
            elif config.DECODER.SIZE == 3:
                # 3*3 conv
                model_decoder = nn.Sequential(
                    nn.Conv2d(in_channels=config.POINT_ENCODER.FEATURE_DIMENSION, out_channels=config.POINT_ENCODER.FEATURE_DIMENSION, kernel_size=3, padding=1),
                    nn.BatchNorm2d(config.POINT_ENCODER.FEATURE_DIMENSION),
                    nn.ReLU()
                )
            else:
                raise Exception(f"Illegal convolution kernel: {config.DECODER.SIZE}")
        else:
            raise Exception(f"Decoder not found: {config.DECODER.NAME}")
    else:
        model_decoder = None


    if config.MODEL.DECODER_COOR:
        if config.DECODER_COOR.NAME.lower() == "conv":
            if model_decoder is not None:
                model_decoder_coor = nn.Conv2d(config.POINT_ENCODER.FEATURE_DIMENSION, config.DECODER_COOR.OUT_CHANNELS, 1)
            else:
                model_decoder_coor = nn.Sequential(
                        nn.Conv2d(in_channels=config.POINT_ENCODER.FEATURE_DIMENSION, out_channels=config.DECODER_COOR.OUT_CHANNELS, kernel_size=3, padding=1),
                        nn.BatchNorm2d(config.DECODER_COOR.OUT_CHANNELS),
                        nn.ReLU()
                    )
        else:
            raise Exception(f"Decoder_coor not found: {config.DECODER_COOR.NAME}")
    else:
        model_decoder_coor = None

    if config.MODEL.DECODER_SEM:
        if config.DECODER_SEM.NAME.lower() == "conv":
            if model_decoder is not None:
                model_decoder_sem = nn.Conv2d(config.POINT_ENCODER.FEATURE_DIMENSION, config.DECODER_SEM.OUT_CHANNELS, 1)
            else:
                model_decoder_sem = nn.Sequential(
                        nn.Conv2d(in_channels=config.POINT_ENCODER.FEATURE_DIMENSION, out_channels=config.DECODER_SEM.OUT_CHANNELS, kernel_size=3, padding=1),
                        nn.BatchNorm2d(config.DECODER_SEM.OUT_CHANNELS),
                        nn.ReLU()
                    )
        else:
            raise Exception(f"Decoder_sem not found: {config.DECODER_SEM.NAME}")
    else:
        model_decoder_sem = None

    # needed by CSBEV
    if config.MODEL.SAM:
        if config.SAM.NAME.lower() == "efficientsam":
            model_SAM = build_efficient_sam_vitt()
            for param in model_SAM.parameters():
                param.requires_grad = False

            if config.CONV == 1:
                # 1*1 conv
                model_adapt_SAM = nn.Sequential(
                    nn.Conv2d(in_channels=config.POINT_ENCODER.FEATURE_DIMENSION, out_channels=3, kernel_size=1),
                    nn.BatchNorm2d(3),
                    nn.ReLU()
                )
            elif config.CONV == 3:
                # 3*3 conv
                model_adapt_SAM = nn.Sequential(
                    nn.Conv2d(in_channels=config.POINT_ENCODER.FEATURE_DIMENSION, out_channels=3, kernel_size=3, padding=1),
                    nn.BatchNorm2d(3),
                    nn.ReLU()
                )
            else:
                raise Exception(f"Illegal convolution kernel: {config.CONV}")
        else:
            raise Exception(f"SAM model not found: {config.SAM.NAME}")
    else:
        model_SAM = None
        model_adapt_SAM = None

    if config.MODEL.TEACHER:
        model_teacher = SECOND(config.POINT_ENCODER.IN_CHANNELS, config.POINT_ENCODER.OUT_CHANNELS, config=config)
        model_teacher = load_pretrain_params_from_file(model_teacher, pretrain_path=config.TEACHER.PATH, logger=logger)
        for param in model_teacher.parameters():
            param.requires_grad = False
    else:
        model_teacher = None

    return model_points, model_decoder, model_decoder_coor, model_decoder_sem, model_SAM, model_adapt_SAM, model_teacher

def load_pretrain_params_from_file(model, pretrain_path=None, to_cpu=False, logger=None):
    if not os.path.isfile(pretrain_path):
        raise FileNotFoundError
    
    logger.info('==> Loading parameters from pretrain model %s to %s' % (pretrain_path, 'CPU' if to_cpu else 'GPU'))
    loc_type = torch.device('cpu') if to_cpu else None
    pretrain_checkpoint = torch.load(pretrain_path, map_location=loc_type)

    pretrain_dict = pretrain_checkpoint['model_points']
    model_dict = model.state_dict()
    update_count = 0
    for k, v in pretrain_dict.items():
        if k in model_dict:
            model_dict.update({k: v})
            update_count += 1
            logger.info('Updated weight %s: %s' % (k, str(pretrain_dict[k].shape)))
        else:
            logger.info('Not updated weight %s: %s' % (k, str(pretrain_dict[k].shape)))
    model.load_state_dict(model_dict)

    logger.info('==> Done (loaded %d/%d)' % (update_count, len(model_dict)))
    
    return model