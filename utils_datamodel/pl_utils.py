import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from utils_datamodel.datasets import Cifar10Corrupted, MelanomaDataset
from torchvision import datasets, transforms
from torch.utils.data import random_split, Subset
import os
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import shutil

__all__ = ['CIFARDataModule', 'SVHNDataModule', 'MNISTDataModule', 'CIFAR10CorruptedDataModule', 'MelanomDataModule']


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, wandb_logger: WandbLogger, ds_name: str, data_dir: str = "datasets", batch_size: int = 32,
                 reuse_artifact=False):
        super().__init__()
        self.data_dir = data_dir
        self.ds_test = None
        self.ds_train = None
        self.ds_val = None
        self.wandb_logger_ = wandb_logger
        self.reuse_artifact_ = reuse_artifact
        self.ds_name_ = ds_name
        ds_dir = os.path.join(self.data_dir, f"{ds_name}_ds")
        self.data_dir_ram = os.path.join("/dev/shm", ds_dir.split('/')[-1])
        self.save_hyperparameters('ds_name', 'data_dir', 'batch_size', 'reuse_artifact')

    @classmethod
    def add_data_module_specific_args(cls, parent_parser):
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument('--ds_name', type=str, default=argparse.SUPPRESS)
        parser.add_argument('--data_dir', type=str, default=argparse.SUPPRESS)
        parser.add_argument('--batch_size', type=int, default=argparse.SUPPRESS)
        parser.add_argument('--reuse_artifact', action='store_true', default=argparse.SUPPRESS)
        return parent_parser

    def get_ds_train_val(self, data_dir: str) -> datasets:
        """
        :param data_dir: root path for dataset (Up on this dir, it creates two sub dirs "{ds_name_}_ds" and "{ds_name_}_split"
        :return: returns a dataset for train and validation data
        """
        raise NotImplementedError("you must implement this method")

    def get_ds_test(self, data_dir: str) -> datasets:
        """

        :param data_dir: root path for dataset (Up on this dir, it creates two sub dirs "{ds_name_}_ds" and "{ds_name_}_split"
        :return:
        """
        raise NotImplementedError("you must implement this method")

    def train_val_split(self, ds_full: datasets) -> (datasets, datasets):
        """

        :param ds_full: full dataset from method get_ds_train_val
        :return: tuple of train and valid dataset
        """
        raise NotImplementedError("you must implement this method")

    def prepare_data(self, version='latest') -> None:
        # download data
        ds_dir = os.path.join(self.data_dir, f"{self.ds_name_}_ds")
        os.makedirs(ds_dir, exist_ok=True)
        if self.reuse_artifact_:
            artifact = self.wandb_logger_.experiment.use_artifact(f"{self.ds_name_}_dataset:latest", type='raw_data')
            artifact_ds_dir = artifact.download(root=ds_dir)
        else:
            ds_full = self.get_ds_train_val(ds_dir)
            ds_test = self.get_ds_test(ds_dir)
            art = wandb.Artifact(f"{self.ds_name_}_dataset", type="raw_data")
            art.add_dir(ds_dir)
            self.wandb_logger_.experiment.log_artifact(art)

        # download split
        split_dir = os.path.join(self.data_dir, f"{self.ds_name_}_split/")
        os.makedirs(split_dir, exist_ok=True)
        if self.reuse_artifact_:
            artifact = self.wandb_logger_.experiment.use_artifact(f"{self.ds_name_}_split:{version}", type='split')
            artifact_split_dir = artifact.download(root=split_dir)

        # load into RAM
        print(f"move data to from {ds_dir} to RAM {self.data_dir_ram}...")
        os.system(f"cp -r {ds_dir} /dev/shm")
        print("data_dir: ", self.data_dir)
        self.wandb_logger_.experiment.config.update(self.hparams)

    def setup(self, stage=None, version="latest"):
        self.ds_test = self.get_ds_test(self.data_dir_ram)
        ds_full = self.get_ds_train_val(self.data_dir_ram)

        split_dir = os.path.join(self.data_dir, f"{self.ds_name_}_split/")
        if self.reuse_artifact_:
            train_idx = torch.load(os.path.join(split_dir, 'train_idx.pt'))
            valid_idx = torch.load(os.path.join(split_dir, 'valid_idx.pt'))
            ds_train = torch.utils.data.Subset(ds_full, train_idx)
            ds_val = torch.utils.data.Subset(ds_full, valid_idx)
        else:
            ds_train, ds_val = self.train_val_split(ds_full)
            torch.save(ds_train.indices, os.path.join(split_dir, "train_idx.pt"))
            torch.save(ds_val.indices, os.path.join(split_dir, "valid_idx.pt"))
            art = wandb.Artifact(f"{self.ds_name_}_split", type="split")
            art.add_dir(split_dir)
            self.wandb_logger_.experiment.log_artifact(art, aliases=[version])

        self.ds_train = ds_train
        self.ds_val = ds_val
        print("DM setup done")

    def train_dataloader(self, batch_size: int = None):
        return DataLoader(self.ds_train,
                          batch_size=self.hparams.batch_size if batch_size is None else batch_size,
                          shuffle=True,
                          num_workers=12,
                          persistent_workers=True,
                          prefetch_factor=1,
                          pin_memory=torch.cuda.is_available())

    def val_dataloader(self, batch_size: int = None):
        return DataLoader(self.ds_val,
                          batch_size=1000 if batch_size is None else batch_size,
                          shuffle=False,
                          num_workers=8,
                          persistent_workers=True,
                          prefetch_factor=1,
                          pin_memory=torch.cuda.is_available())

    def test_dataloader(self, batch_size: int = None):
        return DataLoader(self.ds_test,
                          batch_size=1000 if batch_size is None else batch_size,
                          shuffle=False,
                          num_workers=8,
                          prefetch_factor=1,
                          persistent_workers=True,
                          pin_memory=torch.cuda.is_available())


class CIFARDataModule(BaseDataModule):
    def __init__(self, wandb_logger: WandbLogger, data_dir: str = "datasets/cifar", batch_size: int = 32,
                 transform_test: transforms.Compose = transforms.ToTensor(),
                 transform_train: transforms.Compose = transforms.ToTensor(), reuse_artifact=False,
                 seed_for_split: int = 0):
        super().__init__(wandb_logger=wandb_logger,
                         ds_name="cifar10",
                         data_dir=data_dir,
                         batch_size=batch_size,
                         reuse_artifact=reuse_artifact)
        self.seed_for_split_ = seed_for_split
        self.t_train_ = transform_train
        self.t_generic_ = transform_test

    def get_ds_train_val(self, data_dir: str) -> datasets:
        return datasets.CIFAR10(data_dir, download=True, train=True, transform=self.t_train_)

    def get_ds_test(self, data_dir: str) -> datasets:
        return datasets.CIFAR10(data_dir, download=True, train=False, transform=self.t_generic_)

    def train_val_split(self, ds_full: datasets) -> (datasets, datasets):
        return random_split(ds_full, [49000, 1000], generator=torch.Generator().manual_seed(self.seed_for_split_))


class SVHNDataModule(BaseDataModule):
    def __init__(self, wandb_logger: WandbLogger, data_dir: str = "datasets/svhn", batch_size: int = 32,
                 transform_test: transforms.Compose = transforms.ToTensor(),
                 transform_train: transforms.Compose = transforms.ToTensor(), reuse_artifact=False,
                 seed_for_split: int = 0):
        super().__init__(wandb_logger=wandb_logger,
                         ds_name="svhn",
                         data_dir=data_dir,
                         batch_size=batch_size,
                         reuse_artifact=reuse_artifact)
        self.seed_for_split_ = seed_for_split
        self.t_train_ = transform_train
        self.t_generic_ = transform_test

    def get_ds_train_val(self, data_dir: str) -> datasets:
        return datasets.SVHN(data_dir, download=True, split="train", transform=self.t_train_)

    def get_ds_test(self, data_dir: str) -> datasets:
        return datasets.SVHN(data_dir, download=True, split="test", transform=self.t_generic_)

    def train_val_split(self, ds_full: datasets) -> (datasets, datasets):
        return random_split(ds_full, [70000, 3257], generator=torch.Generator().manual_seed(self.seed_for_split_))


class MNISTDataModule(BaseDataModule):
    def __init__(self, wandb_logger: WandbLogger, data_dir: str = "datasets/mnist", batch_size: int = 32,
                 transform_test: transforms.Compose = transforms.ToTensor(),
                 transform_train: transforms.Compose = transforms.ToTensor(), reuse_artifact=False,
                 seed_for_split: int = 0):
        super().__init__(wandb_logger=wandb_logger,
                         ds_name="mnist",
                         data_dir=data_dir,
                         batch_size=batch_size,
                         reuse_artifact=reuse_artifact)
        self.seed_for_split_ = seed_for_split
        self.t_train_ = transform_train
        self.t_generic_ = transform_test

    def get_ds_train_val(self, data_dir: str) -> datasets:
        return datasets.MNIST(data_dir, download=True, train=True, transform=self.t_train_)

    def get_ds_test(self, data_dir: str) -> datasets:
        return datasets.MNIST(data_dir, download=True, train=False, transform=self.t_generic_)

    def train_val_split(self, ds_full: datasets) -> (datasets, datasets):
        return random_split(ds_full, [59000, 1000], generator=torch.Generator().manual_seed(self.seed_for_split_))


class CIFAR10CorruptedDataModule(pl.LightningDataModule):
    def __init__(self, wandb_logger: WandbLogger, data_dir: str = "datasets/cifar10_C", batch_size: int = 32,
                 transform: transforms.Compose = transforms.ToTensor(), reuse_artifact="latest", corruption=[],
                 intensities=range(1, 6)):
        """

        Args:
            wandb_logger:
            data_dir: data directory
            batch_size:
            transform: image transformation method. Input is numpy array [C,W,B]
            reuse_artifact: reuse artifact from wandb. If None create an artifact else tag of artifact. To use the latest use "latest"
            corruption: elements of corruptions. If you pass an emtpy list [] then we will uses all corruptions defined in CIFAR10_C.CORRUPTIONS
            intensities: specify the intensity. Must be in range (1,5)
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transform
        self.ds_ = None
        self.wandb_logger_ = wandb_logger
        self.reuse_artifact_ = reuse_artifact
        self.data_dir_ram = None
        self.ds_name_ = 'cifar10_c'
        self.corruption_ = corruption
        self.intensities_ = intensities
        ds_dir = os.path.join(self.data_dir, f"{self.ds_name_}_ds")
        self.data_dir_ram = os.path.join("/dev/shm", ds_dir.split('/')[-1])

    def get_ds(self, data_dir: str) -> datasets:
        return Cifar10Corrupted(data_dir,
                                download=True,
                                transform=self.transform,
                                corruption=self.corruption_,
                                intensities=self.intensities_)

    def prepare_data(self) -> None:
        # download data
        ds_dir = os.path.join(self.data_dir, f"{self.ds_name_}_ds")
        os.makedirs(ds_dir, exist_ok=True)
        ds_dir_param = os.path.join(self.data_dir, f"{self.ds_name_}_params")
        os.makedirs(ds_dir_param, exist_ok=True)
        if isinstance(self.reuse_artifact_, str):
            artifact = self.wandb_logger_.experiment.use_artifact(f"{self.ds_name_}_dataset:latest", type='raw_data')
            artifact_ds_dir = artifact.download(root=ds_dir)

            artifact = self.wandb_logger_.experiment.use_artifact(f"{self.ds_name_}_ds_params:{self.reuse_artifact_}",
                                                                  type='numpy')
            artifact_ds_dir = artifact.download(root=ds_dir_param)
            npzfiles = np.load(os.path.join(ds_dir_param, 'params.npz'), allow_pickle=True)
            self.corruption_ = npzfiles['corruption']
            self.intensities_ = npzfiles['intensities']
        else:
            ds_full = self.get_ds(ds_dir)
            art = wandb.Artifact(f"{self.ds_name_}_dataset", type="raw_data")
            art.add_dir(ds_dir)
            self.wandb_logger_.experiment.log_artifact(art)

            np.savez(os.path.join(ds_dir_param, 'params.npz'),
                     corruption=self.corruption_,
                     intensities=self.intensities_)
            art = wandb.Artifact(f"{self.ds_name_}_ds_params",
                                 type="numpy",
                                 metadata=dict(corruption=self.corruption_,
                                               intensities=self.intensities_))
            art.add_dir(ds_dir_param)
            self.wandb_logger_.experiment.log_artifact(art)

        # load into RAM
        print("move data to RAM...")
        os.system(f"cp -r {ds_dir} /dev/shm")
        print("data_dir: ", self.data_dir)

    def setup(self, stage=None):
        self.ds_ = self.get_ds(self.data_dir_ram)
        print("DM setup done")

    def predict_dataloader(self, batch_size: int = None):
        return DataLoader(self.ds_,
                          batch_size=self.batch_size if batch_size is None else batch_size,
                          shuffle=False,
                          num_workers=16,
                          persistent_workers=True,
                          pin_memory=torch.cuda.is_available())


class MelanomDataModule(BaseDataModule):
    def __init__(self, wandb_logger: WandbLogger, data_dir: str = "dataset/siim_isic", batch_size: int = 32,
                 transform_test: transforms.Compose = transforms.ToTensor(),
                 transform_train: transforms.Compose = transforms.ToTensor(), 
                 target_transform=None, reuse_artifact=False,
                 seed_for_split: int = 0,
                 meta_features: list = None,
                 **kwargs):
        """

        :param wandb_logger: wandb logger to log artifacts and hyperparameters
        :param data_dir: local data directory
        :param batch_size:
        :param transform_test: transformation function for test data
        :param transform_train: transformation function for train and valid data
        :param reuse_artifact: if true load data from wandb server. (For the first time it must be false)
        :param seed_for_split: seed to split train, valid and test dataset
        :param meta_features: List of meta features such as 'sex' or 'age_approx' (not every entry supported yet)
        """
        super().__init__(wandb_logger=wandb_logger,
                         ds_name="melanom",
                         data_dir=data_dir,
                         batch_size=batch_size,
                         reuse_artifact=reuse_artifact)
        self.seed_for_split_ = seed_for_split
        self.t_train_ = transform_train
        self.t_generic_ = transform_test
        self.meta_features_ = meta_features
        self.age_approx_mu = None
        self.age_approx_std = None
        self.target_transform = target_transform
        self.save_hyperparameters('seed_for_split', 'meta_features')

    @classmethod
    def add_data_module_specific_args(cls, parent_parser):
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument('--seed_for_split', type=int, default=argparse.SUPPRESS)
        parser.add_argument('--meta_features', nargs='+',
                            default=argparse.SUPPRESS)  # as list '+' => 1 or more '*' => 0 or more
        return super().add_data_module_specific_args(parent_parser)

    def prepare_data(self) -> None:
        if not self.reuse_artifact_:
            # clean up textural data
            ds_dir = os.path.join(self.data_dir, f"{self.ds_name_}_ds")
            df = pd.read_csv(os.path.join(ds_dir, 'ISIC_2020_Training_GroundTruth_v2.csv'))
            # One-hot encoding of anatom_site_general_challenge feature
            dummies = pd.get_dummies(df['anatom_site_general_challenge'], dummy_na=True, dtype=np.uint8, prefix='site')
            df = pd.concat([df, dummies.iloc[:df.shape[0]]], axis=1)
            # Sex features
            df['sex'] = df['sex'].map({'male': 1, 'female': 0})
            df = df.dropna(subset=['sex', 'age_approx'])
            # df['sex'] = df['sex'].fillna(-1)
            # Age features
            self.age_approx_mu = df['age_approx'].mean()
            self.age_approx_std = df['age_approx'].std()
            df['age_approx'] = (df['age_approx'] - self.age_approx_mu) / self.age_approx_std
            # df['age_approx'] /= df['age_approx'].max()
            # df['age_approx'] = df['age_approx'].fillna(0)
            # df['patient_id'] = df['patient_id'].fillna(0)
            df['target'] = df['target'].astype(np.float32)

            # split data into train and test set
            idx_train, idx_test = train_test_split(np.arange(len(df)), random_state=self.seed_for_split_,
                                                   train_size=int(len(df) * 0.80163))
            split_dir = os.path.join(self.data_dir, f"{self.ds_name_}_split/")
            np.save(os.path.join(split_dir, "train_val_idx.npy"), idx_train)
            np.save(os.path.join(split_dir, "test_idx.npy"), idx_test)
            self.test_df_ = df.iloc[idx_test].reset_index(drop=True)
            self.train_df_ = df.iloc[idx_train].reset_index(drop=True)
            super(MelanomDataModule, self).prepare_data()
        else:
            super(MelanomDataModule, self).prepare_data()
            # clean up textural data
            ds_dir = os.path.join(self.data_dir, f"{self.ds_name_}_ds")
            df = pd.read_csv(os.path.join(ds_dir, 'ISIC_2020_Training_GroundTruth_v2.csv'))
            # One-hot encoding of anatom_site_general_challenge feature
            dummies = pd.get_dummies(df['anatom_site_general_challenge'], dummy_na=True, dtype=np.uint8, prefix='site')
            df = pd.concat([df, dummies.iloc[:df.shape[0]]], axis=1)
            # Sex features
            df['sex'] = df['sex'].map({'male': 1, 'female': 0})
            df = df.dropna(subset=['sex', 'age_approx'])
            # df['sex'] = df['sex'].fillna(-1)
            # Age features
            self.age_approx_mu = df['age_approx'].mean()
            self.age_approx_std = df['age_approx'].std()
            df['age_approx'] = (df['age_approx'] - self.age_approx_mu) / self.age_approx_std
            # df['age_approx'] /= df['age_approx'].max()
            # df['age_approx'] = df['age_approx'].fillna(0)
            # df['patient_id'] = df['patient_id'].fillna(0)
            df['target'] = df['target'].astype(np.float32)

            # split data into train and test set
            split_dir = os.path.join(self.data_dir, f"{self.ds_name_}_split/")
            idx_train = np.load(os.path.join(split_dir, 'train_val_idx.npy'))
            idx_test = np.load(os.path.join(split_dir, 'test_idx.npy'))
            self.test_df_ = df.iloc[idx_test].reset_index(drop=True)
            self.train_df_ = df.iloc[idx_train].reset_index(drop=True)

    def get_ds_train_val(self, data_dir: str) -> datasets:
        path = os.path.join(data_dir, 'ISIC_2020_Training_JPEG_128x128')
        return MelanomaDataset(df=self.train_df_, imfolder=path, train=True, transforms=self.t_train_, target_transform=self.target_transform, 
                               meta_features=self.meta_features_)

    def get_ds_test(self, data_dir: str) -> datasets:
        path = os.path.join(data_dir, 'ISIC_2020_Training_JPEG_128x128')
        return MelanomaDataset(df=self.test_df_, imfolder=path, train=True, transforms=self.t_generic_, target_transform=self.target_transform, 
                               meta_features=self.meta_features_)

    def train_val_split(self, ds_full: datasets) -> (datasets, datasets):
        return random_split(ds_full, [21200, 5300], generator=torch.Generator().manual_seed(self.seed_for_split_))

class MelanomDataModuleFromSplit(MelanomDataModule):
    def __init__(self, index_file_name: str = "melanoma_ridx.csv", split: int = 1,
                 **kwargs):
        super().__init__(seed_for_split=0, **kwargs)
        self.save_hyperparameters('index_file_name', 'split')
        self.ridx_filename = index_file_name
        assert split in [1,2,3,4,5,6], "split idx should be between (1,2,...,6)"
        self.split_ = split
        self.idx_df = None

    @classmethod
    def add_data_module_specific_args(cls, parent_parser):
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument('--split', type=int, default=argparse.SUPPRESS)
        parser.add_argument('--index_file', type=str, default=argparse.SUPPRESS)
        parser.add_argument('--meta_features', nargs='+',
                            default=argparse.SUPPRESS)  # as list '+' => 1 or more '*' => 0 or more
        return super(MelanomDataModule, cls).add_data_module_specific_args(parent_parser)

    def prepare_data(self) -> None:
        ds_dir = os.path.join(self.data_dir, f"{self.ds_name_}_ds")
        if not self.reuse_artifact_:
            # copy files for artifact
            shutil.copy(os.path.join(self.data_dir, 'ISIC_2020_Training_GroundTruth_v2.csv'),
                        os.path.join(ds_dir, 'ISIC_2020_Training_GroundTruth_v2.csv'))
            shutil.copy(os.path.join(self.data_dir, 'melanoma_ridx.csv'), os.path.join(ds_dir, 'melanoma_ridx.csv'))
            # clean up textural data
            idx_df = pd.read_csv(os.path.join(ds_dir, self.ridx_filename))
            self.idx_df = idx_df[idx_df['spl'] == self.split_]
            df = pd.read_csv(os.path.join(ds_dir, 'ISIC_2020_Training_GroundTruth_v2.csv'))
            # One-hot encoding of anatom_site_general_challenge feature
            dummies = pd.get_dummies(df['anatom_site_general_challenge'], dummy_na=True, dtype=np.uint8, prefix='site')
            df = pd.concat([df, dummies.iloc[:df.shape[0]]], axis=1)
            # Sex features
            df['sex'] = df['sex'].map({'male': 1, 'female': 0})
            df = df.dropna(subset=['sex', 'age_approx'])
            # Age features
            self.age_approx_mu = df['age_approx'].mean()
            self.age_approx_std = df['age_approx'].std()
            df['age_approx'] = (df['age_approx'] - self.age_approx_mu) / self.age_approx_std
            df['target'] = df['target'].astype(np.float32)

            # split data into train and test set (-1 because R count with 1...n))
            train_idx = self.idx_df[self.idx_df['type'] == 'train']['idx'].to_numpy() - 1
            val_idx = self.idx_df[self.idx_df['type'] == 'val']['idx'].to_numpy() - 1
            test_idx = self.idx_df[self.idx_df['type'] == 'test']['idx'].to_numpy() - 1
            idx_train_val = np.append(train_idx, val_idx)

            split_dir = os.path.join(self.data_dir, f"{self.ds_name_}_split/")
            np.save(os.path.join(split_dir, "train_val_idx.npy"), idx_train_val)
            np.save(os.path.join(split_dir, "test_idx.npy"), test_idx)
            self.test_df_ = df.iloc[test_idx].reset_index(drop=True)
            self.train_df_ = df.iloc[idx_train_val].reset_index(drop=True)
            super(MelanomDataModule, self).prepare_data()
        else:
            super(MelanomDataModule, self).prepare_data(version=f"split_{self.split_}")
            # clean up textural data
            idx_df = pd.read_csv(os.path.join(ds_dir, self.ridx_filename))
            self.idx_df = idx_df[idx_df['spl'] == self.split_]
            df = pd.read_csv(os.path.join(ds_dir, 'ISIC_2020_Training_GroundTruth_v2.csv'))
            # One-hot encoding of anatom_site_general_challenge feature
            dummies = pd.get_dummies(df['anatom_site_general_challenge'], dummy_na=True, dtype=np.uint8, prefix='site')
            df = pd.concat([df, dummies.iloc[:df.shape[0]]], axis=1)
            # Sex features
            df['sex'] = df['sex'].map({'male': 1, 'female': 0})
            df = df.dropna(subset=['sex', 'age_approx'])
            # Age features
            self.age_approx_mu = df['age_approx'].mean()
            self.age_approx_std = df['age_approx'].std()
            df['age_approx'] = (df['age_approx'] - self.age_approx_mu) / self.age_approx_std
            df['target'] = df['target'].astype(np.float32)

            # split data into train and test set
            split_dir = os.path.join(self.data_dir, f"{self.ds_name_}_split/")
            idx_train = np.load(os.path.join(split_dir, 'train_val_idx.npy'))
            idx_test = np.load(os.path.join(split_dir, 'test_idx.npy'))
            self.test_df_ = df.iloc[idx_test].reset_index(drop=True)
            self.train_df_ = df.iloc[idx_train].reset_index(drop=True)

    def setup(self, stage=None):
        super().setup(stage, version=f"split_{self.split_}")

    def train_val_split(self, ds_full: datasets) -> (datasets, datasets):
        train_idx = self.idx_df[self.idx_df['type'] == 'train']['idx'].to_numpy() - 1
        val_idx = self.idx_df[self.idx_df['type'] == 'val']['idx'].to_numpy() - 1
        lt = len(train_idx)
        lv = len(val_idx)
        return Subset(ds_full, np.arange(0, lt)), Subset(ds_full, np.arange(lt, lt + lv))
