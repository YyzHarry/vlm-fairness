import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageFile
from torchvision import transforms

import config

ImageFile.LOAD_TRUNCATED_IMAGES = True


DATASETS = [
    'MIMIC',
    'CheXpert',
    'NIH',
    'PadChest',
    'VinDr'
]
ATTRS = ['sex', 'ethnicity', 'age', 'sex_ethnicity']


def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError(f"Dataset not found: {dataset_name}")
    return globals()[dataset_name]


class CXRDataset:
    INPUT_SHAPE = None       # Subclasses should override
    AVAILABLE_ATTRS = None   # Subclasses should override
    SPLITS = {               # Default, subclasses may override
        'tr': 0,
        'va': 1,
        'te': 2
    }
    EVAL_SPLITS = ['te']     # Default, subclasses may override

    def __init__(self, root, split, metadata, transform):
        df = pd.read_csv(metadata)
        df = df[df["split"] == (self.SPLITS[split])]

        self.idx = list(range(len(df)))
        self.x = df["filename"].astype(str).map(lambda x: os.path.join(root, x)).tolist()
        self.transform_ = transform

    def __getitem__(self, index):
        i = self.idx[index]
        x = self.transform(self.x[i])
        return i, x

    def __len__(self):
        return len(self.idx)


class BaseImageDataset(CXRDataset):

    def __init__(self, metadata, split):
        transform = transforms.Compose([
            transforms.Normalize([101.48761] * 3, [83.43944] * 3),
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC)
        ])
        super().__init__('/', split, metadata, transform)

    def transform(self, x):
        if self.__class__.__name__ in ['MIMIC'] and 'MIMIC-CXR-JPG' in x:
            reduced_img_path = list(Path(x).parts)
            reduced_img_path[-5] = 'downsampled_files'
            reduced_img_path = Path(*reduced_img_path).with_suffix('.png')
            if reduced_img_path.is_file():
                x = str(reduced_img_path.resolve())
        elif self.__class__.__name__ in ['VinDr']:
            reduced_img_path = list(Path(x).parts)
            reduced_img_path[-2] = 'downsampled_files'
            reduced_img_path = Path(*reduced_img_path).with_suffix('.png')
            assert reduced_img_path.is_file()
            x = str(reduced_img_path.resolve())

        img = np.array(Image.open(x))
        # handle PadChest raw data differently
        if self.__class__.__name__ in ['PadChest']:
            img = np.uint8(img / (2 ** 16) * 255)
        img = self.preprocess_image(Image.fromarray(img))
        img = np.expand_dims(np.array(img), axis=0)
        img = np.repeat(img, 3, axis=0)
        img = torch.from_numpy(img).type(torch.float32)
        return self.transform_(img)

    @staticmethod
    def preprocess_image(img, desired_size=320):
        old_size = img.size
        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        img = img.resize(new_size, Image.ANTIALIAS)

        # create a new image and paste the resized on it
        new_img = Image.new('L', (desired_size, desired_size))
        new_img.paste(img, ((desired_size - new_size[0]) // 2,
                            (desired_size - new_size[1]) // 2))
        return new_img


class MIMIC(BaseImageDataset):
    INPUT_SHAPE = (3, 224, 224,)
    AVAILABLE_ATTRS = ['sex', 'age', 'ethnicity', 'sex_ethnicity']
    TASKS = config.TASKS['MIMIC']
    SPLITS = {
        'te': 2
    }

    def __init__(self, data_path, split, predict_attr=False):
        metadata = os.path.join(data_path, "MIMIC-CXR-JPG", 'foundation_fair_meta', "metadata.csv")
        super().__init__(metadata, split)


class CheXpert(BaseImageDataset):
    INPUT_SHAPE = (3, 224, 224,)
    AVAILABLE_ATTRS = ['sex', 'age', 'ethnicity', 'sex_ethnicity']
    TASKS = config.TASKS['CheXpert']

    def __init__(self, data_path, split, predict_attr=False):
        metadata = os.path.join(data_path, "chexpert", 'foundation_fair_meta', "metadata.csv")
        super().__init__(metadata, split)


class NIH(BaseImageDataset):
    INPUT_SHAPE = (3, 224, 224,)
    AVAILABLE_ATTRS = ['sex', 'age']
    TASKS = config.TASKS['NIH']
    SPLITS = {
        'te': 2
    }

    def __init__(self, data_path, split, predict_attr=False):
        metadata = os.path.join(data_path, "ChestXray8", 'foundation_fair_meta', "metadata.csv")
        super().__init__(metadata, split)


class PadChest(BaseImageDataset):
    INPUT_SHAPE = (3, 224, 224,)
    AVAILABLE_ATTRS = ['sex', 'age']
    TASKS = config.TASKS['PadChest']
    SPLITS = {
        'te': 2
    }

    def __init__(self, data_path, split, predict_attr=False):
        metafile = f"metadata{'_all' if predict_attr else ''}.csv"
        metadata = os.path.join(data_path, "PadChest", 'foundation_fair_meta', metafile)
        super().__init__(metadata, split)


class VinDr(BaseImageDataset):
    INPUT_SHAPE = (3, 224, 224,)
    AVAILABLE_ATTRS = ['sex', 'age']
    TASKS = config.TASKS['VinDr']
    SPLITS = {
        'te': 2
    }

    def __init__(self, data_path, split, predict_attr=False):
        metadata = os.path.join(data_path, "vindr-cxr", 'foundation_fair_meta', "metadata_eval.csv")
        super().__init__(metadata, split)
