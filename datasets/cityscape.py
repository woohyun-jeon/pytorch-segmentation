import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


ignore_value = 255

class CityScapeDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.img_dir = glob.glob(os.path.join(data_dir, 'leftImg8bit', split, '*', '*_leftImg8bit.png'))
        self.msk_dir = [img_file.replace('leftImg8bit', 'gtFine')[:-4] + '_labelIds.png' for img_file in self.img_dir]
        self.transform = transform


    def __len__(self):
        return len(self.img_dir)


    def msk_to_cls(self, mask_file):
        class_converter = {-1: ignore_value, 0: ignore_value, 1: ignore_value, 2: ignore_value,
                          3: ignore_value, 4: ignore_value, 5: ignore_value, 6: ignore_value,
                          7: 0, 8: 1, 9: ignore_value, 10: ignore_value, 11: 2, 12: 3, 13: 4,
                          14: ignore_value, 15: ignore_value, 16: ignore_value, 17: 5,
                          18: ignore_value, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                          28: 15, 29: ignore_value, 30: ignore_value, 31: 16, 32: 17, 33: 18}
        mask_converted = np.zeros(mask_file.shape[:2], dtype=np.uint8)

        for class_k, class_v in class_converter.items():
            mask_converted[mask_file == class_k] = class_v

        return mask_converted


    def __getitem__(self, idx):
        img = Image.open(self.img_dir[idx]).convert('RGB')
        img = np.array(img) / 255.
        img = img.astype(np.float32)

        msk = Image.open(self.msk_dir[idx])
        msk = np.array(msk)
        msk = self.msk_to_cls(mask_file=msk)

        if self.transform is not None:
            sample_augmented = self.transform(image=img, mask=msk)
            img, msk = sample_augmented['image'], sample_augmented['mask']

        img = torch.from_numpy(img)
        msk = torch.tensor(msk, dtype=torch.long)

        img = img.permute(2,0,1).contiguous()

        return img, msk