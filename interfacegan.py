import argparse
import os
import random
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from data.single_imglatent_dataset import ImgLatentDataset
from sklearn import svm

'''
from torchvision.utils import save_image
from models.stylegan_model import Generator
stylegan = Generator(256, 512, 8).cuda()
# stylegan.load_state_dict(torch.load('/home/xiamf/cartoon_motion/stylegan2-pytorch/checkpoints/checkpoint_afhq/170000.pt')['g_ema'])
stylegan.load_state_dict(torch.load('/home/xiamf/cartoon_motion/stylegan2-pytorch/checkpoints/checkpoint_tower/220000.pt')['g_ema'])
trunc = stylegan.mean_latent(4096).detach()
def generate_img(latent, randomize_noise=False, return_feats=False):
    img, _, feats = stylegan(
        [latent],
        truncation=0.7,
        truncation_latent=trunc,
        input_is_latent=True,
        randomize_noise=randomize_noise,
    )
    if return_feats:
        return img, feats
    else:
        return img
'''

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def make_dataset(folder):
    paths = []
    for root, _, names in os.walk(folder):
        for name in names:
            if 'npy' not in name or 'real' in name:
                continue
            print(name)
            path = os.path.join(root, name)
            paths.append(path)

    return paths


def sample(sample_list, num):
    if len(sample_list) <= num:
        return sample_list
    return random.sample(sample_list, num)


def main(opt):
    setup_seed(opt.seed)

    # target latents
    tgt_folder = os.path.join(opt.data_dir, opt.tgt_folder)
    tgt_paths = make_dataset(tgt_folder)

    # source latents
    if opt.src_folder is None:
        src_folder = opt.data_dir
        _src_paths = make_dataset(src_folder)
        src_paths = list(set(_src_paths) - set(tgt_paths))
    else:
        src_folder = os.path.join(opt.data_dir, opt.src_folder)
        src_paths = make_dataset(src_folder)
    if opt.tgt_num is not None:
        tgt_paths = sample(tgt_paths, opt.tgt_num)
    tgt_latents = [np.load(tgt_path) for tgt_path in tgt_paths]
    tgt_latents = np.concatenate(tgt_latents, axis=0)
    if opt.src_num is not None:
        src_paths = sample(src_paths, opt.src_num)
    src_latents = [np.load(src_path) for src_path in src_paths]
    src_latents = np.concatenate(src_latents, axis=0)

    n_tgt = len(tgt_latents)
    n_src = len(src_latents)
    print(f'tgt: num: {n_tgt}, shape: {tgt_latents.shape}, '
          f'src: num: {n_src}, shape: {src_latents.shape}\n'
          f'src: {opt.src_folder}, tgt: {opt.tgt_folder}')
    
    train_data = np.concatenate([src_latents, tgt_latents], axis=0)
    train_label = np.concatenate([np.zeros(n_src, dtype=np.int), np.ones(n_tgt, dtype=np.int)], axis=0)
    if opt.svm_train_iter:
        clf = svm.SVC(kernel='linear', max_iter=opt.svm_train_iter)
    else:
        clf = svm.SVC(kernel='linear')
    classifier = clf.fit(train_data, train_label)
    boundary = classifier.coef_.reshape(1, -1).astype(np.float32)
    boundary = boundary / np.linalg.norm(boundary)
    save_name = 'boundary_nosrc.npy' if opt.src_folder is None else f'boundary_{opt.src_folder}.npy'
    np.save(os.path.join(opt.save_dir, save_name), boundary)
    '''
    boundary = torch.from_numpy(boundary).cuda()
    latent = stylegan.style(torch.randn(8, 512).cuda())
    latent_5 = latent + boundary * 5
    latent_10 = latent + boundary * 10
    latent_15 = latent + boundary * 15
    img = generate_img(latent)
    img_5 = generate_img(latent_5)
    img_10 = generate_img(latent_10)
    img_15 = generate_img(latent_15)
    save_image(
        torch.cat([img, img_5, img_10, img_15], dim=0),
        'demo.png',
        nrow=8,
        normalize=True,
        range=(-1, 1),
    )
    '''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tgt_folder', type=str, default='white_dog', help='target latent code folder')
    parser.add_argument('--src_folder', type=str, default=None, help='source latent code folder, if set to `None`, then use all other latent codes')
    parser.add_argument('--data_dir', type=str, default='/home/xiamf/Editnet/datasets/celeba_data', help='path to latent code')
    parser.add_argument('--boundary_dir', type=str, default='/home/xiamf/Editnet/interfacegan_boundaries', help='path to save boundaries')
    parser.add_argument('--svm_train_iter', type=int, default=None)
    parser.add_argument('--src_num', type=int, default=30)
    parser.add_argument('--tgt_num', type=int, default=30)
    parser.add_argument('--seed', type=int, default=0)

    opt = parser.parse_args()
    opt.save_dir = os.path.join(opt.boundary_dir, opt.tgt_folder)
    if not os.path.exists(opt.save_dir):
        os.mkdir(opt.save_dir)

    main(opt)