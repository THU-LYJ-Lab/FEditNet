import os
import torch
import numpy as np
from tqdm import tqdm
from data import build_dataloader
from torchvision.utils import save_image
from models.stylegan_model import Generator
from models.models import PatchSampleF
from models.patchnce import PatchNCELoss
from utils.parser import parse_args
from utils.loss_dicts import LossDicts, LossDict, ratio


def main(opt, mode='generate', save=False):
    stylegan =Generator(**opt.stylegan_kwargs).eval().cuda()
    F = PatchSampleF(**opt.F_kwargs).eval().cuda()
    nce_layers = opt.nce_layers
    allowed_keys = [f'l{i}' for i in nce_layers]
    criterionNCE = []
    for _ in nce_layers:
        criterionNCE.append(PatchNCELoss(opt).eval().cuda())

    stylegan_ckpt = torch.load(opt.stylegan_ckpt)
    stylegan.load_state_dict(stylegan_ckpt['g_ema'])
    truncation = opt.truncation
    trunc = stylegan.mean_latent(4096).detach()

    # load ckpt
    ckpt = torch.load(opt.ckpt)
    boundary = torch.nn.functional.normalize(ckpt['G']['boundary'], dim=1).cuda()

    def generate_img(latent=None):
        if latent is None:
            latent = stylegan.style(torch.randn(1, 512).cuda())
        img, _, feats = stylegan(
            [latent],
            truncation=truncation,
            truncation_latent=trunc,
            input_is_latent=True,
            randomize_noise=False,
            layers=nce_layers
        )
        return img, feats, latent

    def calc_nce_loss(_feat_src, _feat_tgt) -> LossDict:
        n_layers = len(nce_layers)
        feat_src, feat_tgt = [], []
        for (k, _src), (_, _tgt) in zip(_feat_src.items(), _feat_tgt.items()):
            if k not in nce_layers:
                continue
            feat_src.append(_src)
            feat_tgt.append(_tgt)
        feat_k_pool, sample_ids = F(feat_src, opt.num_patches, None)
        feat_q_pool, _ = F(feat_tgt, opt.num_patches, sample_ids)

        nce_loss = LossDict()
        for layer, f_q, f_k, crit in zip(nce_layers, feat_q_pool, feat_k_pool, criterionNCE):
            loss = crit(f_q, f_k)
            nce_loss[f'l{layer}'] = loss.mean().detach().cpu().item()

        return nce_loss

    def analyze_once(no_nce_loss=False, latent=None):
        all_imgs = []
        all_nce_losses = LossDicts(allowed_keys=allowed_keys)
        with torch.no_grad():
            img, feats, latent = generate_img(latent)
            all_imgs.append(img)
            for length in range(5, 20, 5):
                latent_edited = latent + boundary * length
                img_edited, feats_edited, _ = generate_img(latent_edited)
                all_imgs.append(img_edited)
                if no_nce_loss:
                    continue
                nce_loss = calc_nce_loss(feats, feats_edited)
                all_nce_losses.add_dict(f'length_{length}', nce_loss)
        return torch.cat(all_imgs), all_nce_losses, latent

    # initialize F, then load state dict
    analyze_once()
    F.load_state_dict(ckpt['F'])

    # dataset, using selected ones
    if mode == 'generate':
        for i in tqdm(range(200)):
            all_imgs, _, latent = analyze_once(no_nce_loss=True)
            save_image(all_imgs,
                       f'./datasets/analysis_data/target/{i}.jpg',
                       nrow=1,
                       normalize=True,
                       range=(-1, 1))
            np.save(f'./datasets/analysis_data/target/{i}.npy',
                    latent.detach().cpu().numpy())
    elif mode == 'analyze':
        losses = LossDicts(allowed_keys=allowed_keys)
        dataloader = build_dataloader(opt, opt.seed)
        for i, data in enumerate(dataloader):
            latent = data['latent'].cuda().view(1, 512)
            _, all_nce_losses, _ = analyze_once(latent=latent)
            # print(f'\n{i}.jpg:')
            # print(all_nce_losses)
            losses = losses + all_nce_losses
            if save:
                np.save(os.path.join(opt.img_dir, f'{i}_nce_loss.npy'), all_nce_losses)
        losses = losses * (1. / len(dataloader.dataset))
        return losses
    elif mode == 'eval':
        losses = LossDicts(allowed_keys=allowed_keys)
        dataloader = build_dataloader(opt, opt.seed)
        for i, data in enumerate(dataloader):
            latent = data['latent'].cuda().view(1, 512)
            all_imgs, all_nce_losses, _ = analyze_once(latent=latent)
            losses = losses + all_nce_losses
            if save:
                np.save(os.path.join(opt.img_dir, f'{i}_nce_loss.npy'), all_nce_losses)
            '''
            save_image(all_imgs,
                       f'./{i}.jpg',
                       nrow=1,
                       normalize=True,
                       range=(-1, 1))
            '''
        losses = losses * (1. / len(dataloader.dataset))
        return losses

            

if __name__ == '__main__':
    args = parse_args()
    print('generate')
    no_change = main(args, 'generate')
    print('real')
    real = main(args, 'eval')
    print(real)
    
    # print('no_change')
    # args.img_dir = './datasets/analysis_data/no_change'
    # args.latent_dir = './datasets/analysis_data/no_change'
    # no_change = main(args, 'analyze')

    print('change')
    args.img_dir = './datasets/analysis_data/target'
    args.latent_dir = './datasets/analysis_data/target'
    change = main(args, 'analyze')

    # print(ratio(change, no_change))
    print(ratio(change, real))
    # print(ratio(no_change, real))
    
