from time import time
import torch
import argparse
import torch.nn.functional as F
import os
import numpy as np
from models.stylegan_model import Generator
from torchvision.utils import save_image, make_grid

'''
opt = parse_args()
datadir = '/home/xiamf/stylegan_disentanglement/celeba_data/train_Smile'
datalist = [name for name in os.listdir(datadir) if 'npy' in name]
latents = []
for name in datalist:
    latentdir = os.path.join(datadir, name)
    latent = torch.Tensor(np.load(latentdir))
    latents.append(latent)

latents = torch.cat(latents, dim=0).cuda()
'''

def generate_img(latent, g_ema, trunc):
    img, _, _ = g_ema(
        [latent],
        truncation=0.7,
        truncation_latent=trunc,
        input_is_latent=True,
        randomize_noise=False,
    )
    return img


def check_ckpt_name(ckpt_name, ckpt_freq):
    if '.pt' not in ckpt_name:
        return False
    n_iter = int(ckpt_name[:-3])
    return (n_iter % ckpt_freq) == 0


def test_folder(g_ema, trunc, folder_path, ckpt_freq):
    ckpt_list = [name for name in os.listdir(folder_path) if check_ckpt_name(name, ckpt_freq)]
    ckpt_list = sorted(ckpt_list, key=lambda x: int(x[:-3]))[:7]
    boundaries = [F.normalize(torch.load(os.path.join(folder_path, ckpt_path))['G']['boundary'], dim=1).cuda() for ckpt_path in ckpt_list]
    save_path = folder_path.replace('checkpoints', 'test')
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    for i in range(50):
        with torch.no_grad():
            latent = torch.randn(1, 512).cuda()
            latent = g_ema.style(latent)
            img = generate_img(latent, g_ema, trunc).expand(len(boundaries), -1, -1, -1)
            latent_editeds_5 = [latent + boundary * 5 for boundary in boundaries]
            latent_editeds_10 = [latent + boundary * 10 for boundary in boundaries]
            latent_editeds_15 = [latent + boundary * 15 for boundary in boundaries]
            latent_editeds_20 = [latent + boundary * 20 for boundary in boundaries]
            img_edited_5 = torch.cat([generate_img(latent_edited_5, g_ema, trunc) for latent_edited_5 in latent_editeds_5], dim=0)
            img_edited_10 = torch.cat([generate_img(latent_edited_10, g_ema, trunc) for latent_edited_10 in latent_editeds_10], dim=0)
            img_edited_15 = torch.cat([generate_img(latent_edited_15, g_ema, trunc) for latent_edited_15 in latent_editeds_15], dim=0)
            img_edited_20 = torch.cat([generate_img(latent_edited_20, g_ema, trunc) for latent_edited_20 in latent_editeds_20], dim=0)
            imgs = torch.cat([img, img_edited_5, img_edited_10, img_edited_15, img_edited_20], 0)
            save_image(imgs, f'{save_path}/{i}.jpg', nrow=len(boundaries), normalize=True, range=(-1, 1))
    return ckpt_list


def test(g_ema, trunc, result_folder, skip_words=None, require_words=None, ckpt_freq=5000):
    _ckpt_folders = os.listdir(result_folder)
    if skip_words is not None:
        _ckpt_folders = [name for name in _ckpt_folders if not any(skip_word in name for skip_word in skip_words)]
    if require_words is not None:
        _ckpt_folders = [name for name in _ckpt_folders if not any(require_word not in name for require_word in require_words)]
    ckpt_folders = sorted(_ckpt_folders)
    print(ckpt_folders)

    for folder in ckpt_folders:
        folder_path = os.path.join(result_folder, folder)
        test_folder(g_ema, trunc, folder_path, ckpt_freq)

'''
ckpt_list = ['/home/xiamf/Editnet/checkpoints_smiles/nce3_fix_frozenD/012000.pt',
             '/home/xiamf/Editnet/checkpoints/smile5shot_nce3_fix_frozenD/012000.pt']

boundaries = [F.normalize(torch.load(ckpt_path)['G']['boundary'], dim=1).cuda() for ckpt_path in ckpt_list]

for i in range(50):
    with torch.no_grad():
        latent = editnet.generate_latent(10)
        img = editnet.generate_img(latent)
        latent_editeds_5 = [latent + boundary * 5 for boundary in boundaries]
        latent_editeds_10 = [latent + boundary * 10 for boundary in boundaries]
        latent_editeds_15 = [latent + boundary * 15 for boundary in boundaries]
        img_edited_5 = torch.cat([editnet.generate_img(latent_edited_5) for latent_edited_5 in latent_editeds_5], dim=0)
        img_edited_10 = torch.cat([editnet.generate_img(latent_edited_10) for latent_edited_10 in latent_editeds_10], dim=0)
        img_edited_15 = torch.cat([editnet.generate_img(latent_edited_15) for latent_edited_15 in latent_editeds_15], dim=0)
        print(img.shape, img_edited_5.shape, img_edited_10.shape)
        save_image(torch.cat([img, img_edited_5, img_edited_10, img_edited_15], 0), f'samples_smiles/test/{i}.jpg', nrow=10, normalize=True, range=(-1, 1))
'''  

parser = argparse.ArgumentParser()

parser.add_argument('--result_folder', type=str, default='checkpoints_smiles')
parser.add_argument('--require_words', type=str, default=None)
parser.add_argument('--skip_words', type=str, default=None)
parser.add_argument('--stylegan_ckpt', type=str, default='/home/xiamf/cartoon_motion/stylegan2-pytorch/checkpoints/checkpoint_celeba/740000.pt')
parser.add_argument('--img_size', type=int, default=512)
parser.add_argument('--ckpt_freq', type=int, default=5000)

opt = parser.parse_args()
g_ema = Generator(opt.img_size, 512, 8).cuda()
stylegan_ckpt = torch.load(opt.stylegan_ckpt)
g_ema.load_state_dict(stylegan_ckpt['g_ema'])
trunc = g_ema.mean_latent(4096).detach()
result_folder = opt.result_folder
require_words = None if opt.require_words is None else opt.require_words.split(',')
skip_words = None if opt.skip_words is None else opt.skip_words.split(',')
test(g_ema, trunc, result_folder, require_words=require_words, skip_words=skip_words, ckpt_freq=opt.ckpt_freq)

