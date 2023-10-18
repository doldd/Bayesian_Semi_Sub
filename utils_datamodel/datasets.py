import numpy as np
import torch
import torchvision.datasets as datasets
import os
from sklearn.model_selection import KFold
import zipfile
import urllib.request

import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
from torchvision.datasets.utils import download_and_extract_archive, check_integrity
import os
import pandas as pd


class Cifar10Corrupted(Dataset):
    """ Represent the corrupted CIFAR10 dataset alongside correct labels"""
    CORRUPTIONS = ['brightness', 'defocus_blur', 'elastic_transform',
                   'fog', 'frost', 'gaussian_blur', 'gaussian_noise',
                   'glass_blur', 'jpeg_compression', 'saturate',
                   'shot_noise', 'snow', 'spatter',
                   'speckle_noise', 'zoom_blur', 'motion_blur',
                   'contrast', 'impulse_noise', 'pixelate']
    CIFAR10_TEST_N = 10000

    tgz_md5 = '56bf5dcef84df0e2308c6dcbcbbd8499'
    url = 'https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1'
    filename = 'CIFAR-10-C.tar'
    fileList = [['frost.npy', '31f6ab3bce1d9934abfb0cc13656f141'],
                ['elastic_transform.npy', '9421657c6cd452429cf6ce96cc412b5f'],
                ['motion_blur.npy', 'fffa5f852ff7ad299cfe8a7643f090f4'],
                ['labels.npy', 'c439b113295ed5254878798ffe28fd54'],
                ['impulse_noise.npy', '2090e01c83519ec51427e65116af6b1a'],
                ['gaussian_blur.npy', 'c33370155bc9b055fb4a89113d3c559d'],
                ['brightness.npy', '0a81ef75e0b523c3383219c330a85d48'],
                ['glass_blur.npy', '7361fb4019269e02dbf6925f083e8629'],
                ['saturate.npy', '1cfae0964219c5102abbb883e538cc56'],
                ['shot_noise.npy', '3a7239bb118894f013d9bf1984be7f11'],
                ['fog.npy', '7b397314b5670f825465fbcd1f6e9ccd'],
                ['spatter.npy', '8a5a3903a7f8f65b59501a6093b4311e'],
                ['gaussian_noise.npy', 'ecaf8b9a2399ffeda7680934c33405fd'],
                ['contrast.npy', '3c8262171c51307f916c30a3308235a8'],
                ['zoom_blur.npy', '6ea8e63f1c5cdee1517533840641641b'],
                ['jpeg_compression.npy', '2b9cc4c864e0193bb64db8d7728f8187'],
                ['defocus_blur.npy', '7d1322666342a0702b1957e92f6254bc'],
                ['speckle_noise.npy', 'ef00b87611792b00df09c0b0237a1e30'],
                ['snow.npy', 'bb238de8555123da9c282dea23bd6e55'],
                ['pixelate.npy', '0f14f7e2db14288304e1de10df16832f']]
    base_folder = 'CIFAR-10-C'

    def __init__(self, data_dir, corruption: list = [], intensities=range(1, 6), transform=None, download=True):
        """
        Initialise the dataset and load data to memory
        Parameters
        ------
        - data_dir (str): path directory to the dataset.
        - corruptions (List[str]): if corrupted, which corruptions to load (None to load all)
        - intensities (List[int]): the intensities to load (1 to 5) for the specfied corruptions
        """
        self.labels = None
        self.corrupted_images = None
        self.root = data_dir
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.load_corrupted_cifar(os.path.join(self.root, self.base_folder),
                                  corruption if len(corruption) > 0 else self.CORRUPTIONS,
                                  intensities)
        self.transform = transform

    def __len__(self):
        return self.corrupted_images.shape[0]

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in self.fileList:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def __getitem__(self, idx):
        img = self.corrupted_images[idx]
        label = self.labels[idx % self.CIFAR10_TEST_N]
        if self.transform:
            img = self.transform(img)
        return img, label

    def load_corrupted_cifar(self, data_dir, corruptions, intensities):
        """
        Load data with relevant corruptions to memory
        Parameters
        ------
        - data_dir (str): path directory to the dataset.
        - corruptions (List[str]): if corrupted, which corruptions to load
        - intensities (List[int]): the intensities to load (1 to 5) for the specfied corruptions
        """
        self.labels = np.load(os.path.join(data_dir, 'labels.npy')).astype(np.int64)

        sets = []
        for corruption in corruptions:
            imgs = np.load(f'{data_dir}/{corruption}.npy')
            for intensity in intensities:
                sets.append(imgs[(intensity - 1) * self.CIFAR10_TEST_N: intensity * self.CIFAR10_TEST_N])
        self.corrupted_images = np.concatenate(sets, axis=0)


class MelanomaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, imfolder: str, train: bool = True, transforms = None, meta_features = None, target_transform = None):
        """
        Class initialization
        Args:
            df (pd.DataFrame): DataFrame with data description
            imfolder (str): folder with images
            train (bool): flag of whether a training dataset is being initialized or testing one
            transforms: image transformation method to be applied
            meta_features (list): list of features with meta information, such as sex and age

        """
        self.df = df
        self.imfolder = os.path.join(imfolder, 'train') if train else os.path.join(imfolder, 'test')
        self.transforms = transforms
        self.target_transform = target_transform
        self.train = train
        self.meta_features = meta_features

    def __getitem__(self, index):
        im_path = os.path.join(self.imfolder, self.df.iloc[index]['image_name'] + '.jpg')
        x = torchvision.io.read_image(im_path)
        if self.transforms:
            x = self.transforms(x)
        data = x
        if self.meta_features is not None:
            meta = np.array(self.df.iloc[index][self.meta_features].values, dtype=np.float32)
            data = (x, meta)

        if self.train:
            y = self.df.iloc[index]['target']
            if self.target_transform:
                y = self.target_transform(y)
            return data, y
        else:
            return data

    def __len__(self):
        return len(self.df)
    
""" 
script adapted from https://gist.github.com/martinferianc/db7615c85d5a3a71242b4916ea6a14a2 
"""
class UCIDatasets():
    def __init__(self,  name,  data_path="", n_splits = 10):
        self.datasets = {
            "housing": "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
            "concrete": "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls",
            "energy": "http://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx",
            "power": "https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip",
            "wine": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
            "yacht": "http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data",
            "airfoil": "https://archive.ics.uci.edu/static/public/291/airfoil+self+noise.zip"}
        self.data_path = data_path
        self.name = name
        self.n_splits = n_splits
        self._load_dataset()

    
    def _load_dataset(self):
        if self.name not in self.datasets:
            raise Exception("Not known dataset!")
        if not os.path.exists(self.data_path+"UCI"):
            os.mkdir(self.data_path+"UCI")

        url = self.datasets[self.name]
        file_name = url.split('/')[-1]
        if not os.path.exists(self.data_path+"UCI/" + file_name):
            urllib.request.urlretrieve(
                self.datasets[self.name], self.data_path+"UCI/" + file_name)
        data = None

        if self.name == "housing":
            data = pd.read_csv(self.data_path+'UCI/housing.data',
                        header=0, delimiter="\s+").values
            self.data = data[np.random.permutation(np.arange(len(data)))]

        elif self.name == "concrete":
            data = pd.read_excel(self.data_path+'UCI/Concrete_Data.xls',
                               header=0).values
            self.data = data[np.random.permutation(np.arange(len(data)))]
        elif self.name == "energy":
            data = pd.read_excel(self.data_path+'UCI/ENB2012_data.xlsx',
                                 header=0).values
            self.data = data[np.random.permutation(np.arange(len(data)))]
        elif self.name == "power":
            zipfile.ZipFile(self.data_path +"UCI/CCPP.zip").extractall(self.data_path +"UCI/CCPP/")
            data = pd.read_excel(self.data_path+'UCI/CCPP/Folds5x2_pp.xlsx', header=0).values
            np.random.shuffle(data)
            self.data = data
        elif self.name == "wine":
            data = pd.read_csv(self.data_path + 'UCI/winequality-red.csv',
                               header=1, delimiter=';').values
            self.data = data[np.random.permutation(np.arange(len(data)))]

        elif self.name == "yacht":
            data = pd.read_csv(self.data_path + 'UCI/yacht_hydrodynamics.data',
                               header=1, delimiter='\s+').values
            self.data = data[np.random.permutation(np.arange(len(data)))]

        elif self.name == "airfoil":
            zipfile.ZipFile(self.data_path +"UCI/airfoil+self+noise.zip").extractall(self.data_path +"UCI/airfoil+self+noise/")
            with open(self.data_path +'UCI/airfoil+self+noise/airfoil_self_noise.dat', 'r') as dat_file, open(self.data_path +'UCI/airfoil+self+noise/airfoil_self_noise.dat', 'w') as csv_file:
                for line in dat_file:
                    csv_file.write(line.replace('\t', ';'))
            data = pd.read_csv(self.data_path+'UCI/airfoil+self+noise/airfoil_self_noise.csv', header=0, delimiter=';').values
            self.data = data[np.random.permutation(np.arange(len(data)))]
            
        kf = KFold(n_splits=self.n_splits)
        self.in_dim = data.shape[1] - 1
        self.out_dim = 1
        self.data_splits = kf.split(data)
        self.data_splits = [(idx[0], idx[1]) for idx in self.data_splits]
    
    def get_split(self, split=-1, train=True):
        if split == -1:
            split = 0
        if 0<=split and split<self.n_splits: 
            train_index, test_index = self.data_splits[split]
            x_train, y_train = self.data[train_index,
                                    :self.in_dim], self.data[train_index, self.in_dim:]
            x_test, y_test = self.data[test_index, :self.in_dim], self.data[test_index, self.in_dim:]
            x_means, x_stds = x_train.mean(axis=0), x_train.var(axis=0)**0.5
            y_means, y_stds = y_train.mean(axis=0), y_train.var(axis=0)**0.5
            x_train = (x_train - x_means)/x_stds
            y_train = (y_train - y_means)/y_stds
            x_test = (x_test - x_means)/x_stds
            y_test = (y_test - y_means)/y_stds
            if train:
                inps = torch.from_numpy(x_train).float()
                tgts = torch.from_numpy(y_train).float()
                train_data = torch.utils.data.TensorDataset(inps, tgts)
                return train_data
            else:
                inps = torch.from_numpy(x_test).float()
                tgts = torch.from_numpy(y_test).float()
                test_data = torch.utils.data.TensorDataset(inps, tgts)
                return test_data