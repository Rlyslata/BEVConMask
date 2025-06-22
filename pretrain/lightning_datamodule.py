import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pretrain.dataloader_nuscenes import NuscenesDataset, CollateSpconv

# from memory_profiler import profile

class PretrainDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dataset = config.DATASET
        self.batch_size = config.OPTIMIZATION.BATCH_SIZE_PER_GPU
        self.num_workers = config.OPTIMIZATION.NUM_WORKERS_PER_GPU

    def setup(self, stage):
        # Dataset
        if self.dataset.NAME.lower() == "nuscenes":
            Dataset = NuscenesDataset
        else:
            raise Exception("Dataset Unknown")
        
        if self.dataset.DATA_SPLIT['train'] in ("parametrize", "parametrizing"):
            phase_train = "parametrizing"
            phase_val = "verifying"
        else:
            phase_train = "train"
            phase_val = "val"

        # Train dataset
        self.train_dataset = Dataset(
            phase=phase_train,
            config=self.config,
            shuffle=True,
        )
        print("Dataset Loaded")
        print("training size: ", len(self.train_dataset))

        # Val dataset
        if self.dataset.NAME.lower() == "nuscenes":
            self.val_dataset = Dataset(
                phase=phase_val,
                config=self.config,
                shuffle=False,
                cached_nuscenes=self.train_dataset.nusc,
            )
        print("validation size: ", len(self.val_dataset))

    # @profile(stream=open('./CSBEV/output/memory/train_dataloader.log', 'w+'))
    def train_dataloader(self):

        if self.dataset.NAME.lower() == "nuscenes":
            default_collate_pair_fn = CollateSpconv(self.config).collate_spconv

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=default_collate_pair_fn,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=lambda id: np.random.seed(
                torch.initial_seed() // 2 ** 32 + id
            ),
        )

    def val_dataloader(self):

        if self.dataset.NAME.lower() == "nuscenes":
            default_collate_pair_fn = CollateSpconv(self.config).collate_spconv

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=default_collate_pair_fn,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=lambda id: np.random.seed(
                torch.initial_seed() // 2 ** 32 + id
            ),
        )
