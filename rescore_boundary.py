import os
import argparse
import random
import torch
import face_alignment
import torch.nn as nn
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


RESCORE_CKPT_PATHS = {
    'Bangs': './checkpoints/resnet_predictor/Bangs_10.pt',
    'Male': './checkpoints/resnet_predictor/Male_10.pt',
    'Smiling': './checkpoints/resnet_predictor/Smiling_10.pt',
    'Young': './checkpoints/resnet_predictor/Young_10.pt',
}


RESCORE_PREDICTORS = dict()
for name, path in RESCORE_CKPT_PATHS.items():
    predictor = resnet50(2).eval().cuda()
    predictor.load_state_dict(torch.load(path))
    RESCORE_PREDICTORS[name] = predictor


def calc_yaw(img, detector):
    img_numpy = img[0].detach().cpu().numpy().transpose([1, 2, 0])
    img_numpy = (img_numpy + 1) * 127.5
    lms_68 = detector.get_landmarks(img_numpy)[0]
    left_eye = (lms_68[36] + lms_68[39] ) / 2
    right_eye = (lms_68[42] + lms_68[45]) / 2
    target = np.array([0., 1.])
    angle1 = np.array([right_eye[0]-left_eye[0], right_eye[2]-left_eye[2]])
    leng1 = np.linalg.norm(angle1)
    angle1 = np.arcsin(angle1.dot(target) / leng1)
    return angle1


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
            if all([ext not in name for ext in _CKPT_EXTENSIONS]):
                continue
            if skip_words is not None:
                if any([word in name for word in skip_words]):
                    continue
            if num_freq is not None:
                n_iter = int(name[:-3])
                if n_iter % num_freq != 0:
                    continue
            ckpt_name = os.path.join(root, name)
            if skip_words is not None:
                if any([word in ckpt_name for word in skip_words]):
                    continue
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
                                 None, opt.skip_words)
        ckpt_list = sorted(ckpt_list)
        named_boundaries.update(dict(zip(ckpt_list, get_interface_boundaries(ckpt_list))))

    boundaries = list(named_boundaries.values())
    num_boundary = len(boundaries)
    assert num_boundary > 0, (f'The number of boundaries to test should be positive, '
                              f'however # boundaries = {num_boundary} received.')

    lengths = range(5, 20, 5)
    set_seed(opt.seed)
    stylegan = Generator(opt.img_size, 512, 8).cuda().eval()
    stylegan_ckpt = torch.load(opt.stylegan_ckpt)['g_ema']
    stylegan.load_state_dict(stylegan_ckpt)
    trunc = stylegan.mean_latent(4096).detach()

    detector = face_alignment.FaceAlignment(
            landmarks_type=face_alignment.LandmarksType._3D,
            device='cuda')

    # test
    # first record the order of the images
    with open(os.path.join(save_dir, 'rescore_boundaries.txt'), 'w') as fp:
        info = f'num: {opt.num}\n'
        info += f'num_boundary: {num_boundary}\n'
        info += f'path to boundaries:\n\t'
        info += '\n\t'.join(list(named_boundaries.keys())) + '\n'
        fp.write(info)
        print(info)

    with torch.no_grad():
        scores = [{length: {rescore_name: 0. for rescore_name in RESCORE_CKPT_PATHS} for length in lengths} for _ in boundaries]

        for idx in tqdm(range(opt.num), total=opt.num):
            latent = stylegan.style(torch.randn(1, 512).cuda())
            img_ori = generate_img(latent, stylegan, trunc)
            
            # scores_origin = dict(Pose=calc_yaw(img_ori, detector))
            scores_origin = dict()
            for rescore_name in RESCORE_CKPT_PATHS:
                _, score_origin = RESCORE_PREDICTORS[rescore_name](nn.Upsample(128)(img_ori))
                scores_origin[rescore_name] = score_origin

            for i, boundary in enumerate(boundaries):
                for length in lengths:
                    latent_edited = latent + boundary * length
                    img_edited = generate_img(latent_edited, stylegan, trunc)
                    
                    # scores[i][length]['Pose'] = calc_yaw(img_edited, detector) - scores_origin['Pose']
                    for rescore_name in RESCORE_CKPT_PATHS:
                        _, score_edited = RESCORE_PREDICTORS[rescore_name](nn.Upsample(128)(img_edited))
                        scores[i][length][rescore_name] += (score_edited - scores_origin[rescore_name])[:, 1].sum()

        info = 'rescore:\n'
        for i in range(num_boundary):
            info += f'{list(named_boundaries.keys())[i]}:\n'
            for length in lengths:
                info += f'length {length}:\n'
                for rescore_name in scores[i][length]:
                    rescore = scores[i][length][rescore_name] / float(opt.num)
                    info += f'rescore predictor {rescore_name}: {rescore:.4f}\n'

        with open(os.path.join(save_dir, 'rescore_boundaries.txt'), 'a+') as fp:
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
    parser.add_argument('--num', type=int, default=2000)

    opt = parser.parse_args()
    if opt.skip_words is not None:
        opt.skip_words = opt.skip_words.split(',')
    if opt.require_words is not None:
        opt.require_words = opt.require_words.split(',')
    main(opt)
