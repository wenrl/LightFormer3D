import os
# gpu = '0'
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu
import sys
# sys.path.append("/root/data-tmp/MIS3D/EffiDec3D/brats_dataset/")
from sklearn.model_selection import KFold
import os
import json
import math
import numpy as np
import torch
from monai import transforms, data
import SimpleITK as sitk
from tqdm import tqdm
from torch.utils.data import Dataset
import random
random.seed(1111)
import torch
import torch.nn.parallel
import torch.utils.data.distributed

from monai.data import DataLoader

import argparse



def get_dataloader(dataset, shuffle=False, batch_size=1):
    if dataset is None:
        return None

    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=4)

class PretrainDataset(Dataset):
    def __init__(self, datalist, transform=None, cache=False) -> None:
        super().__init__()
        self.transform = transform
        self.datalist = datalist
        self.cache = cache
        if cache:
            self.cache_data = []
            for i in tqdm(range(len(datalist)), total=len(datalist)):
                d = self.read_data(datalist[i])
                self.cache_data.append(d)

    def read_data(self, data_path):

        file_identifizer = data_path.split("/")[-1].split("_")[-1]
        image_paths = [
            os.path.join(data_path, f"BraTS2021_{file_identifizer}_t1.nii.gz"),
            os.path.join(data_path, f"BraTS2021_{file_identifizer}_flair.nii.gz"),
            os.path.join(data_path, f"BraTS2021_{file_identifizer}_t2.nii.gz"),
            os.path.join(data_path, f"BraTS2021_{file_identifizer}_t1ce.nii.gz")
        ]
        seg_path = os.path.join(data_path, f"BraTS2021_{file_identifizer}_seg.nii.gz")

        image_data = [sitk.GetArrayFromImage(sitk.ReadImage(p)) for p in image_paths]
        seg_data = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))

        image_data = np.array(image_data).astype(np.float32)
        seg_data = np.expand_dims(np.array(seg_data).astype(np.int32), axis=0)
        return {
            "image": image_data,
            "label": seg_data
        }

    def __getitem__(self, i):
        if self.cache:
            image = self.cache_data[i]
        else:
            try:
                image = self.read_data(self.datalist[i])
            except:
                with open("./bugs.txt", "a+") as f:
                    f.write(f"error，{self.datalist[i]}\n")
                if i != len(self.datalist) - 1:
                    return self.__getitem__(i + 1)
                else:
                    return self.__getitem__(i - 1)
        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.datalist)


def get_loader_brats(data_dir, batch_size=1, fold=0, num_workers=8):
    all_dirs = os.listdir(data_dir)
    all_paths = [os.path.join(data_dir, d) for d in all_dirs]
    
    random.shuffle(all_paths)
    size = len(all_paths)
    train_size = int(0.7 * size)
    val_size = int(0.1 * size)
    train_files = all_paths[:train_size]
    val_files = all_paths[train_size:train_size + val_size]
    test_files = all_paths[train_size + val_size:]
    print(f"train is {len(train_files)}, val is {len(val_files)}, test is {len(test_files)}")

    # train_transform = transforms.Compose(
    #     [
    #         transforms.ConvertToMultiChannelBasedOnBratsClassesD(keys=["label"]),
    #         transforms.CropForegroundd(keys=["image", "label"], source_key="image"),

    #         # transforms.RandSpatialCropd(keys=["image", "label"], roi_size=[96, 96, 96],
    #                                     # random_size=False),
    #         # transforms.SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96)),
    #         transforms.SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96)),
        
    #         # 随机裁剪到固定大小96×96×96
    #         transforms.RandSpatialCropd(
    #             keys=["image", "label"], 
    #             roi_size=[96, 96, 96],
    #             random_size=False  # 固定大小裁剪
    #             ),
    #         transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    #         transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    #         transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    #         transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),

    #         transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
    #         transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    #         transforms.ToTensord(keys=["image", "label"], ),
    #     ]
    # )
    # val_transform = transforms.Compose(
    #     [transforms.ConvertToMultiChannelBasedOnBratsClassesD(keys=["label"]),
    #      transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
    #      # 确保验证集也裁剪到96×96×96
    #      transforms.SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96)),
    #      transforms.CenterSpatialCropd(keys=["image", "label"], roi_size=(96, 96, 96)),
    #      transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    #      transforms.ToTensord(keys=["image", "label"]),
    #      ]
    # )
    train_transform = transforms.Compose(
        [   
            transforms.ConvertToMultiChannelBasedOnBratsClassesD(keys=["label"]),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),

            transforms.RandSpatialCropd(keys=["image", "label"], roi_size=[96, 96, 96],
                                        random_size=False),
            transforms.SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96)),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            transforms.ToTensord(keys=["image", "label"],),
        ]
    )
    val_transform = transforms.Compose(
        [   transforms.ConvertToMultiChannelBasedOnBratsClassesD(keys=["label"]),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),

            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    train_ds = PretrainDataset(train_files, transform=train_transform)

    val_ds = PretrainDataset(val_files, transform=val_transform)
    test_ds = PretrainDataset(test_files, transform=val_transform)

    loader = [train_ds, val_ds, test_ds]

    return loader

if __name__ == "__main__":
    data_dir = ".dataset//BraTS2021"


    env = "pytorch"
    max_epoch = 300
    batch_size = 2
    val_every = 30
    num_gpus = 1
    train_ds, val_ds, test_ds = get_loader_brats(data_dir=data_dir, batch_size=batch_size)
    train_loader = get_dataloader(train_ds, shuffle=True, batch_size=self.batch_size)
    if val_ds is not None:
        val_loader = get_dataloader(val_ds, shuffle=False, batch_size=1)
    if test_ds is not None:
        test_loader = get_dataloader(test_ds, shuffle=False, batch_size=1)
