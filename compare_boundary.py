import os
import argparse
import random
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.utils import save_image
from models.models import resnet50
from models.stylegan_model import Generator
from models.inception_model import build_inception_model
from utils.utils import set_seed, compute_fid_from_feature


_CKPT_EXTENSIONS = ['.npy', '.pth', '.pt']


def get_editnet_boundaries(ckpt_list):
    boundaries = []
    for ckpt_path in ckpt_list:
        ckpt = torch.load(ckpt_path)
        boundary = ckpt['G']['boundary']
        boundaries.append(F.normalize(boundary, dim=1).cuda())
    return boundaries


def get_interface_boundaries(ckpt_list):
    boundaries = []
    for ckpt_path in ckpt_list:
        boundary = np.load(ckpt_path)
        boundaries.append(torch.from_numpy(boundary).cuda())
    return boundaries


@torch.no_grad()
def generate_img(latent, stylegan, trunc):
    img, _, _ = stylegan(
        [latent],
        truncation=0.7,
        truncation_latent=trunc,
        input_is_latent=True,
        randomize_noise=False,
    )
    return img


def make_dataset(base_dir, target, require_words, skip_words, num_freq=None):
    ckpt_list = []
    for root, _, names in os.walk(base_dir):
        folder = os.path.basename(root)
        if target not in folder.lower():
            continue
        if require_words is not None:
            if any([word not in folder for word in require_words]):
                continue
        if skip_words is not None:
            if any([word in folder for word in skip_words]):
                continue
        for name in names:
            if skip_words is not None:
                if any([word in name for word in skip_words]):
                    continue
            if all([ext not in name for ext in _CKPT_EXTENSIONS]):
                continue
            if num_freq is not None:
                n_iter = int(name[:-3])
                if n_iter % num_freq != 0:
                    continue
            ckpt_name = os.path.join(root, name)
            ckpt_list.append(ckpt_name)
    return ckpt_list


def main(opt):
    # saving settings
    save_dir = os.path.join(opt.save_dir, opt.target)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # traverse all potential boundaries
    named_boundaries = dict()
    if opt.editnet_dir is not None:
        ckpt_list = make_dataset(opt.editnet_dir, opt.target,
                                 opt.require_words, opt.skip_words,
                                 opt.editnet_iter)
        ckpt_list = sorted(ckpt_list)
        named_boundaries.update(dict(zip(ckpt_list, get_editnet_boundaries(ckpt_list))))

    if opt.interface_dir is not None:
        ckpt_list = make_dataset(opt.interface_dir, opt.target,
                                 opt.require_words, opt.skip_words)
        ckpt_list = sorted(ckpt_list)
        named_boundaries.update(dict(zip(ckpt_list, get_interface_boundaries(ckpt_list))))

    boundaries = list(named_boundaries.values())
    num_boundary = len(boundaries)
    assert num_boundary > 0, (f'The number of boundaries to test should be positive, '
                              f'however # boundaries = {num_boundary} received.')
    # shape of (bs * num_boundary, 512), which is repeated as
    # (w, x, y, z, w, x, y, z) when num_boundary = 4 and batch_size = 2,
    # where w, x, y, z are the boundaries
    boundaries = torch.cat(boundaries, dim=0).repeat(opt.batch_size, 1)

    # basic settings
    # lengths = range(5, 25, 5)
    lengths = range(5, 25, 5)
    set_seed(opt.seed)
    stylegan = Generator(opt.img_size, 512, 8).cuda().eval()
    stylegan_ckpt = torch.load(opt.stylegan_ckpt)['g_ema']
    stylegan.load_state_dict(stylegan_ckpt)
    trunc = stylegan.mean_latent(4096).detach()

    inception_model = build_inception_model(
                                    opt.inception_ckpt).cuda().eval()

    # test
    # first record the order of the images
    with open(os.path.join(save_dir, 'named_boundaries.txt'), 'w') as fp:
        info = f'num: {opt.num}\n'
        info += f'num_boundary: {num_boundary}\n'
        info += f'batch_size: {opt.batch_size}\n'
        info += f'path to boundaries:\n\t'
        info += '\n\t'.join(list(named_boundaries.keys())) + '\n'
        fp.write(info)
        print(info)

    n_row = opt.batch_size * num_boundary

    with torch.no_grad():
        cache_path = opt.cache_path
        real_features = np.load(cache_path)
        fake_features = [{length: [] for length in lengths}  for _ in range(num_boundary)]
        fids = [{length: 0. for length in lengths}  for _ in range(num_boundary)]

        for idx in tqdm(range(opt.num)):
            latent = stylegan.style(torch.randn(opt.batch_size, 512).cuda())
            # shape of (bs * num_boundary, 512), which is repeated as
            # (a, a, a, a, b, b, b, b) when num_boundary = 4 and batch_size = 2,
            # where a, b are the genereted latent code
            latent = latent.repeat(1, num_boundary).view(-1, 512)
            img_ori = generate_img(latent, stylegan, trunc)
            all_images = [img_ori]

            for length in lengths:
                latent_edited = latent + boundaries * length
                img_edited = generate_img(latent_edited, stylegan, trunc)
                fake_feature = inception_model(img_edited)
                fake_feature = fake_feature.view(opt.batch_size, num_boundary, -1)
                fake_feature = fake_feature.transpose(0, 1)
                for i in range(num_boundary):
                    fake_features[i][length].append(fake_feature[i])
                all_images.append(img_edited)

            imgs = torch.cat(all_images, dim=0)
            
            save_image(imgs,
                       os.path.join(save_dir, f'{idx}.jpg'),
                       nrow=n_row,
                       normalize=True,
                       range=(-1, 1))
        '''
        info = 'fid score:\n'
        for i in range(num_boundary):
            info += f'{list(named_boundaries.keys())[i]}:\n'
            for length in lengths:
                fake_features[i][length] = torch.cat(fake_features[i][length], 0).detach().cpu().numpy()
                fid = compute_fid_from_feature(fake_features[i][length], real_features)
                fids[i][length] = fid
                info += f'length {length}: {fid:.4f}\n'
        '''

        with open(os.path.join(save_dir, 'named_boundaries.txt'), 'a+') as fp:
            fp.write(info)
            print(info)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument('--seed', type=int, default=0)

    # StyleGAN settings
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--stylegan_ckpt', type=str, default='/home/xiamf/cartoon_motion/stylegan2-pytorch/checkpoints/checkpoint_celeba/740000.pt', help='path to checkpoint of pretrained StyleGAN2')
    # parser.add_argument('--stylegan_ckpt', type=str, default='/home/xiamf/cartoon_motion/stylegan2-pytorch/checkpoints/checkpoint_afhq/170000.pt', help='path to checkpoint of pretrained StyleGAN2')
    # parser.add_argument('--stylegan_ckpt', type=str, default='/home/xiamf/cartoon_motion/stylegan2-pytorch/checkpoints/checkpoint_tower/220000.pt', help='path to checkpoint of pretrained StyleGAN2')
    # parser.add_argument('--stylegan_ckpt', type=str, default='/home/xiamf/cartoon_motion/stylegan2-pytorch/checkpoints/checkpoint_church/270000.pt', help='path to checkpoint of pretrained StyleGAN2')

    # boundary directions
    parser.add_argument('--editnet_dir', type=str, default=None)
    parser.add_argument('--editnet_iter', type=int, default=5000)
    parser.add_argument('--interface_dir', type=str, default=None)
    parser.add_argument('--inception_ckpt', type=str, default='./checkpoints/pt_inception-2015-12-05-6726825d.pth')

    # target attribute
    parser.add_argument('--target', type=str, default='black', help='target attribute to edit')
    parser.add_argument('--cache_path', type=str, default='./datasets/afhq_data/tiger/real_features.npy')
    parser.add_argument('--require_words', type=str, default=None)
    parser.add_argument('--skip_words', type=str, default=None)

    # saving settings
    parser.add_argument('--save_dir', type=str, default='compare')

    # test settings
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num', type=int, default=200)

    opt = parser.parse_args()
    if opt.skip_words is not None:
        opt.skip_words = opt.skip_words.split(',')
    if opt.require_words is not None:
        opt.require_words = opt.require_words.split(',')
    main(opt)
