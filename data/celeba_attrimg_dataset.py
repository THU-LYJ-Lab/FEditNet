import os
import random
import torch
import numpy as np
import pandas as pd
import PIL.Image as Image
import torch.utils.data as data
import torchvision.transforms as transforms


_LATENT_EXTENSIONS = ['npy']
_IMG_EXTENSIONS = ['jpg', 'png']


def latent_check(name):
    return '.' in name and name.split('.')[-1] in _LATENT_EXTENSIONS and 'real_feature' not in name


def img_check(name):
    return '.' in name and name.split('.')[-1] in _IMG_EXTENSIONS


def make_dataset(folder, data_type):
    paths = []
    assert os.path.isdir(folder), f'{folder} is not a valid directory!'
    check = img_check if data_type == 'img' else latent_check
    for root, _, fnames in sorted(os.walk(folder)):
        for fname in fnames:
            if not check(fname):
                continue
            path = os.path.join(root, fname)
            paths.append(path)
    paths = sorted(paths, key=lambda path: int(os.path.basename(path).split('.')[0]))
    return paths


class AttrImgDataset(data.Dataset):
    def __init__(self, 
                 data_dir='./datasets/celeba/CelebA-HQ-img',
                 attr_dir='./datasets/celeba/CelebAMask-HQ-attribute-anno.txt',
                 attr_pairs=None,
                 transform=None
                 ):
        self.data_dir = data_dir
        self.attr_dir = attr_dir
        self.attr_pairs = attr_pairs
        self.img_paths = make_dataset(self.data_dir, 'img')
        self.length = len(self.img_paths)
        self.transform = transform

        if not os.path.exists(attr_dir):
            raise FileExistsError(f'{attr_dir} not exists!')
        attr = pd.read_csv(attr_dir, delim_whitespace=True, header=1)
        self.attr_anno = list(attr)
        attr = torch.as_tensor(attr.values)
        self.attr = (attr + 1) // 2  # map from {-1, 1} to {0, 1}

        if attr_pairs is not None:
            if any(map(lambda x: x not in self.attr_anno, attr_pairs)):
                raise ValueError(f'Got wrong attributes: {attr_pairs}, allowed '
                                 f'attribute list: {self.attr_anno}!')
            self.attr_indices = list(map(lambda x: self.attr_anno.index(x), self.attr_pairs))
            # print(self.attr_indices)

    def __getitem__(self, index):
        img_path = self.img_paths[index % self.length]
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        if self.attr_pairs is not None:
            label = self.attr[index][self.attr_indices]
        else:
            label = self.attr[index]
        # print(index, self.attr[index], img_path, label)
        return dict(img=img, label=label, img_path=img_path)

    def __len__(self):
        return self.length

    def get_indices(self, attr):
        if not isinstance(attr, list):
            attr = [attr]
        return list(map(lambda x: self.attr_anno.index(x), attr))

