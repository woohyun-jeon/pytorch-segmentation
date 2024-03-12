import os
import numpy as np
import pandas as pd
from skimage import io
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class_map = {'VenusExpress': 0, 'Cheops': 1, 'LisaPathfinder': 2, 'ObservationSat1': 3, 'Proba2': 4,
             'Proba3': 5, 'Proba3ocs': 6, 'Smart1': 7, 'Soho': 8, 'XMM Newton': 9}


class AI4SpaceDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = os.path.join(data_dir)
        self.labels = pd.read_csv(os.path.join(data_dir, f'{split}.csv'))
        self.class_map = class_map
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def msk_to_cls(self, mask_file, class_number):
        class_converter = {(0,0,0): 0, (254,0,0): class_number, (0,0,254): 255}
        mask_converted = np.zeros(mask_file.shape[:2], dtype=np.uint8)

        for class_k, class_v in class_converter.items():
            mask_converted[(mask_file == class_k).all(axis=2)] = class_v

        return mask_converted

    def __getitem__(self, idx):
        sat_name = self.labels.iloc[idx]['Class']
        img_name = self.labels.iloc[idx]['Image name']
        msk_name = self.labels.iloc[idx]['Mask name']

        img_file = f'{self.data_dir}/images/{sat_name}/{self.split}/{img_name}'
        msk_file = f'{self.data_dir}/mask/{sat_name}/{self.split}/{msk_name}'

        img, msk = io.imread(img_file), io.imread(msk_file)
        msk = self.msk_to_cls(mask_file=msk, class_number=self.class_map[sat_name])

        if self.transform is not None:
            sample_augmented = self.transform(image=img, mask=msk)
            img, msk = sample_augmented['image'], sample_augmented['mask']

        # convert img, msk to PyTorch tensor
        img = transforms.ToTensor()(img)
        msk = torch.tensor(msk, dtype=torch.long)

        return img, msk