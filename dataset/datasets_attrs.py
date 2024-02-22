import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageFile
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


DATASETS = [
    'MIMICAttr',
    'CheXpertAttr',
    'NIHAttr',
    'PadChestAttr',
    'VinDrAttr'
]
ATTRS = ['sex', 'ethnicity', 'age', 'sex_ethnicity']


class CXRAttrDataset:
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
        df = df[df['split'] == (self.SPLITS[split])]

        df['a'] = df[self.attr_name]
        self.idx = list(range(len(df)))
        self.x = df['filename'].astype(str).map(lambda x: os.path.join(root, x)).tolist()
        self.a = df['a'].astype(int).tolist()
        self.transform_ = transform

    def __getitem__(self, index):
        i = self.idx[index]
        x = self.transform(self.x[i])
        a = torch.tensor(self.a[i], dtype=torch.long)
        return i, x, a

    def __len__(self):
        return len(self.idx)


class BaseImageAttrDataset(CXRAttrDataset):

    def __init__(self, metadata, split, attr_name):
        transform = transforms.Compose([
            transforms.Normalize([101.48761] * 3, [83.43944] * 3),
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC)
        ])
        self.attr_name = attr_name
        super().__init__('/', split, metadata, transform)

    def transform(self, x):
        if self.__class__.__name__ in ['MIMICAttr'] and 'MIMIC-CXR-JPG' in x:
            reduced_img_path = list(Path(x).parts)
            reduced_img_path[-5] = 'downsampled_files'
            reduced_img_path = Path(*reduced_img_path).with_suffix('.png')
            if reduced_img_path.is_file():
                x = str(reduced_img_path.resolve())
        elif self.__class__.__name__ in ['VinDrAttr']:
            reduced_img_path = list(Path(x).parts)
            reduced_img_path[-2] = 'downsampled_files'
            reduced_img_path = Path(*reduced_img_path).with_suffix('.png')
            assert reduced_img_path.is_file()
            x = str(reduced_img_path.resolve())

        img = np.array(Image.open(x))
        # handle PadChest raw data differently
        if self.__class__.__name__ in ['PadChestAttr']:
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


class MIMICAttr(BaseImageAttrDataset):
    INPUT_SHAPE = (3, 224, 224,)
    AVAILABLE_ATTRS = ['sex', 'age', 'ethnicity', 'sex_ethnicity']

    def __init__(self, data_path, split, attr):
        metadata = os.path.join(data_path, "MIMIC-CXR-JPG", 'foundation_fair_meta', "metadata_attr_lr.csv")
        super().__init__(metadata, split, attr)


class CheXpertAttr(BaseImageAttrDataset):
    INPUT_SHAPE = (3, 224, 224,)
    AVAILABLE_ATTRS = ['sex', 'age', 'ethnicity', 'sex_ethnicity']

    def __init__(self, data_path, split, attr):
        metadata = os.path.join(data_path, "chexpert", 'foundation_fair_meta', "metadata_attr_lr.csv")
        super().__init__(metadata, split, attr)


class NIHAttr(BaseImageAttrDataset):
    INPUT_SHAPE = (3, 224, 224,)
    AVAILABLE_ATTRS = ['sex', 'age']

    def __init__(self, data_path, split, attr):
        metadata = os.path.join(data_path, "ChestXray8", 'foundation_fair_meta', "metadata_attr_lr.csv")
        super().__init__(metadata, split, attr)


class PadChestAttr(BaseImageAttrDataset):
    INPUT_SHAPE = (3, 224, 224,)
    AVAILABLE_ATTRS = ['sex', 'age']

    def __init__(self, data_path, split, attr):
        metadata = os.path.join(data_path, "PadChest", 'foundation_fair_meta', "metadata_attr_lr.csv")
        super().__init__(metadata, split, attr)


class VinDrAttr(BaseImageAttrDataset):
    INPUT_SHAPE = (3, 224, 224,)
    AVAILABLE_ATTRS = ['sex', 'age']

    def __init__(self, data_path, split, attr):
        metadata = os.path.join(data_path, "vindr-cxr", 'foundation_fair_meta', "metadata_attr_lr.csv")
        super().__init__(metadata, split, attr)
