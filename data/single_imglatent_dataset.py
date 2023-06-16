"""The class for dataset in EditNet."""
import os
import random
import torch
import cv2
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms


_LATENT_EXTENSIONS = ['npy']
_IMG_EXTENSIONS = ['jpg', 'png']


def load_img(img_path, size):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    img[:,:,:3] = img[:,:,:3][:,:,::-1]
    img = img.astype(np.float32) / 127.5 - 1
    img = img.transpose((2, 0, 1)).clip(-1, 1)
    return torch.from_numpy(img).unsqueeze(0)


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

    return paths


def sample(l, indices):
    return list(map(lambda idx: l[idx], indices))


class ImgLatentDataset(data.Dataset):
    """The class for loading paired latent codes and images."""
    def __init__(self, opt):
        self.opt = opt
        self.img_dir = opt.img_dir
        self.latent_dir = opt.latent_dir
        self.img_paths = sorted(make_dataset(self.img_dir, 'img'))
        self.latent_paths = sorted(make_dataset(self.latent_dir, 'latent'))
        if len(self.img_paths) != len(self.latent_paths):
            raise ValueError(f'The images and latent codes should be paired, '
                             f'i.e, `len(self.img_paths)` and '
                             f'`len(self.latent_paths)` should be the same, '
                             f'however, ({len(self.img_paths)}, '
                             f'{len(self.latent_paths)}) is received!')
        self.length = len(self.img_paths)

        if opt.num_shot is not None:
            indices = list(range(self.length))
            if opt.num_shot < self.length:
                indices = random.sample(indices, opt.num_shot)
                
            self.img_paths = sample(self.img_paths, indices)
            self.latent_paths = sample(self.latent_paths, indices)

        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ]
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        img_path = self.img_paths[index % len(self.img_paths)]
        latent_path = self.latent_paths[index % len(self.latent_paths)]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (self.opt.img_size, self.opt.img_size),
                         interpolation=cv2.INTER_CUBIC)
        img[:,:,:3] = img[:,:,:3][:,:,::-1]
        img = img.astype(np.float32) / 127.5 - 1
        img = img.transpose((2, 0, 1)).clip(-1, 1)
        latent = np.load(latent_path)

        return dict(latent=latent.squeeze(0), img=img,
                    latent_path=latent_path, img_path=img_path)

    def __len__(self):
        return len(self.img_paths)
