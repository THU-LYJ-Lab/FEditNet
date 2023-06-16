import os
import numpy as np
import torch
from torchvision.utils import save_image
from models.stylegan_model import Generator
from models.inception_model import build_inception_model

def generate_img(latent):
    img, _, _ = g(
        [latent],
        truncation=0.7,
        truncation_latent=trunc,
        input_is_latent=True,
    )
    return img


def read_npy(path):
    array = np.load(path)
    return torch.from_numpy(array).cuda()

# latent_dir = '/home/xiamf/Editnet/datasets/celeba_data'
# ckpt = '/home/xiamf/cartoon_motion/stylegan2-pytorch/checkpoints/checkpoint_celeba/740000.pt'

# latent_dir = '/home/xiamf/Editnet/datasets/afhq_data'
# ckpt = '/home/xiamf/cartoon_motion/stylegan2-pytorch/checkpoints/checkpoint_afhq/170000.pt'

# latent_dir = '/home/xiamf/Editnet/datasets/LSUN_data/church'
# ckpt = '/home/xiamf/cartoon_motion/stylegan2-pytorch/checkpoints/checkpoint_church/270000.pt'

# latent_dir = '/home/xiamf/Editnet/datasets/LSUN_data/tower'
# ckpt = '/home/xiamf/cartoon_motion/stylegan2-pytorch/checkpoints/checkpoint_tower/220000.pt'

latent_dir = '/home/xiamf/Editnet/datasets/ffhq_data'
ckpt = '/home/xiamf/cartoon_motion/stylegan2-pytorch/checkpoints/checkpoint_ffhq/390000.pt'

img_size = 512
# img_size = 256

inception_ckpt = '/home/xiamf/Editnet/checkpoints/pt_inception-2015-12-05-6726825d.pth'
g = Generator(img_size, 512, 8).eval().cuda()
g.load_state_dict(torch.load(ckpt)['g_ema'])
trunc = g.mean_latent(4096)

inception_model = build_inception_model(inception_ckpt).eval().cuda()

with torch.no_grad():
    for folder, _, names in os.walk(latent_dir):
        real_features = []
        for name in names:
            if 'npy' not in name:
                continue
            if 'real_feature' in name:
                continue
            img_name = name.replace('npy', 'jpg')
            latent = read_npy(os.path.join(folder, name))
            img = generate_img(latent)
            real_feature = inception_model(img)
            real_features.append(real_feature)
            save_image(img, os.path.join(folder, img_name), normalize=True, range=(-1, 1))
            
        if len(real_features) == 0:
            continue
        print(folder, len(real_features))
        real_features = torch.cat(real_features, dim=0).detach().cpu().numpy()
        cache_filename = 'real_features.npy'
        cache_path = os.path.join(folder, cache_filename)
        np.save(cache_path, real_features)
