import os
import argparse
import torch.nn as nn
import pytorch_lightning as pl
from pathlib import Path
from datetime import datetime as dt
from utils.logger import make_logger
from utils.config import generate_config, log_config
from pretrain_mae.model_builder import make_model
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint
from pretrain_mae.lightning_trainer import LightningPretrain
from pretrain_mae.lightning_datamodule import PretrainDataModule

def main():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument('--name', type=str, default='default', help='name of the experiment')
    parser.add_argument('--config_file', type=str, default='/opt/data/private/CSBEV/config_mae/text_point_point/mae_mse.yaml', help='specify the config for training')
    parser.add_argument("--resume_path", type=str, default=None, help="provide a path to resume an incomplete training")
    parser.add_argument("--save_path", type=str, default=None, help="provide a path to save model")
    parser.add_argument("--num_epochs", type=int, default=None, help="epochs")
    parser.add_argument("--bev_name", type=str, default='ng1_lb-2.0', help="bev name")
    parser.add_argument("--box_point_name", type=str, default='ng1_lb-1_ms3_f10', help="box and point name")
    parser.add_argument("--conv", type=int, default=3, help="model adapter: convolution kernel size")
    # parser.add_argument("--bev_mae_name", type=str, default='hight', help="bev mae name")
    parser.add_argument("--mask_sem", type=float, default=0.6, help="mask ratio semantic")
    parser.add_argument("--mask_non_sem", type=float, default=0.6, help="mask ratio non semantic")
    parser.add_argument("--exc_road", action='store_true', default=False, help='semantic except road')
    parser.add_argument("--mask_with_sam", action='store_true', default=False, help='mask with sam')
    parser.add_argument("--prompt", type=str, default='box', help='box prompt is used by default')
    parser.add_argument("--smooth", type=float, default=0.0, help='using label smoothing')
    args = parser.parse_args()

    config = generate_config(args.config_file)
    if args.resume_path:
        config.MODEL.RESUME_PATH = args.resume_path
    if args.save_path:
        config.MODEL.SAVE_PATH = args.save_path
    if args.num_epochs:
        config.OPTIMIZATION.NUM_EPOCHS = args.num_epochs
    if args.bev_name:
        config.BEV_NAME = args.bev_name
    if args.box_point_name:
        config.BOX_POINT_NAME = args.box_point_name
    if args.conv:
        config.CONV = args.conv
    # if args.bev_mae_name:
    #     config.BEV_MAE_NAME = args.bev_mae_name
    if args.mask_sem:
        config.DATASET.MASK_RATIO_SEMANTIC = args.mask_sem
    if args.mask_non_sem:
        config.DATASET.MASK_RATIO_NON_SEMANTIC = args.mask_non_sem
    config.DATASET.EXCEPT_ROAD = args.exc_road
    config.DATASET.MASK_WITH_SAM = args.mask_with_sam
    config.DATASET.PROMPT = args.prompt
    config.SMOOTH = args.smooth
    
    config.MODEL.SAVE_PATH = Path(config.MODEL.SAVE_PATH, args.name, dt.today().strftime("%d%m%y-%H%M"))
    ckpt = config.MODEL.SAVE_PATH / 'ckpt'
    ckpt.mkdir(parents=True, exist_ok=True)
    if os.environ.get("LOCAL_RANK", 0) == 0:
        log_file = config.MODEL.SAVE_PATH / 'log_train.txt'
        logger = make_logger(log_file, 0)
        logger.info("==============Logging config==============")
        log_config(config, logger)

    dm = PretrainDataModule(config)
    model_points, model_decoder, model_decoder_coor, model_decoder_sem, model_SAM, model_adapt_SAM, model_teacher  = make_model(config, logger)

    module = LightningPretrain(model_points, model_decoder, model_decoder_coor, model_decoder_sem, model_SAM, model_adapt_SAM, model_teacher, config)

    model_checkpoint = ModelCheckpoint(dirpath=ckpt, 
                                       filename='{epoch}-{train_loss:.2f}', 
                                       save_top_k=-1, 
                                       save_on_train_epoch_end=True,
                                       )
    trainer = pl.Trainer(
        gpus=config.OPTIMIZATION.NUM_GPU,
        accelerator="ddp",
        default_root_dir=config.MODEL.SAVE_PATH,
        checkpoint_callback=True,
        max_epochs=config.OPTIMIZATION.NUM_EPOCHS,
        plugins=DDPPlugin(find_unused_parameters=True),
        num_sanity_val_steps=0,
        resume_from_checkpoint=config.MODEL.RESUME_PATH,
        check_val_every_n_epoch=10,
        callbacks=[model_checkpoint],
    )
    print("Starting the training")
    start_time = dt.now().timestamp()
    trainer.fit(module, dm)
    end_time = dt.now().timestamp()
    different_time = end_time - start_time
    print(f"Training time: {int(different_time//3600)} hours, {int(different_time//60 - different_time//3600 * 60)} minutes, {int(different_time%60)} seconds")


if __name__ == "__main__":
    main()
