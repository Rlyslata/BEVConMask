from model import (
    SECOND,
)
from efficient_sam.build_efficient_sam import build_efficient_sam_vitt

import torch.nn as nn

def make_model(config):
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

    return model_points, model_SAM, model_adapt_SAM