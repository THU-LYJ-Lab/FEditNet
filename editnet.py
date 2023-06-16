"""Class for Editnet."""
import os
from copy import deepcopy
from math import fabs
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
from models.models import resnet50
from models.models import PatchSampleF, BoundaryGenerator
from models.stylegan_model import Discriminator, Generator
from models.inception_model import build_inception_model
from models.augment import AugmentPipe
from models.patchnce import PatchNCELoss
from utils.utils import compute_fid_from_feature
from utils.bayes_opt import suggest_weights


def normalize(tensor):
    if tensor.dim() == 2:
        return F.normalize(tensor, dim=1)
    if tensor.dim() == 3:
        return F.normalize(tensor, dim=[1,2])
    raise ValueError(f'The dimension of input tensor should be '
                     f'2 or 3, but {tensor.dim()} received!')


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


class EditNet():
    """Class for Editnet."""
    def __init__(self, opt):
        # training parameters
        self.opt = deepcopy(opt)
        self.latent_dim = opt.latent_dim
        self.img_size = opt.img_size
        self.batch_size = opt.batch_size
        self.nce_layers = opt.nce_layers
        self.learnable_weights = opt.learnable_weights

        # names for networks, losses and visuals
        # models
        self.model_names = ['G', 'D', 'F']

        # D losses
        self.loss_names = ['d']
        # G losses
        self.loss_names += ['g', 'patchnce']

        # visuals
        self.visual_names = ['syn_img', 'syn_img_edited']

        # learnable loss weights
        # rescores predictors
        self.pos_rescore_predictors = []
        self.neg_rescore_predictors = []
        self.pos_rescore_names = []
        self.neg_rescore_names = []
        if self.learnable_weights:
            for ckpt_name in self.opt.pos_rescore_ckpts:
                self.pos_rescore_names.append(os.path.basename(ckpt_name))
                predictor = resnet50(2).eval().cuda()
                predictor.load_state_dict(torch.load(ckpt_name))
                self.pos_rescore_predictors.append(predictor)
            for ckpt_name in self.opt.neg_rescore_ckpts:
                self.neg_rescore_names.append(os.path.basename(ckpt_name))
                predictor = resnet50(2).eval().cuda()
                predictor.load_state_dict(torch.load(ckpt_name))
                self.neg_rescore_predictors.append(predictor)

        # loss weights
        if self.learnable_weights:
            self.optim_weights_infos = dict()
            self.nce_loss_weights = torch.ones(len(self.nce_layers)).cuda()
            self.weights_list = []
            self.pos_rescores_list = []
            self.neg_rescores_list = []
            self.targets_list = []

        # define networks
        # conditional boundary, used to help to keep some features fixed
        self.cond_boundary = None

        # boundary generator
        self.G = BoundaryGenerator(**opt.G_kwargs).cuda()
        # network for nce loss
        self.F = PatchSampleF(**opt.F_kwargs).cuda()
        # pretrained StyleGAN
        self.stylegan = Generator(**opt.stylegan_kwargs).cuda()
        # discriminator
        self.D = Discriminator(**opt.D_kwargs).cuda()
        # augmentations
        self.augment_pipe = AugmentPipe(**opt.aug_kwargs).cuda()
        self.augment_pipe.train().requires_grad_(False)
        self.augment_pipe.p.copy_(torch.as_tensor(1.))

        # inception models to calculate fid
        self.inception_model = build_inception_model(
                                    opt.inception_ckpt).cuda().eval()

        # load pretrained StyleGAN2 model
        stylegan_ckpt = torch.load(self.opt.stylegan_ckpt)
        self.D.load_state_dict(stylegan_ckpt['d'])
        self.stylegan.load_state_dict(stylegan_ckpt['g_ema'])
        self.truncation = self.opt.truncation
        self.trunc = self.stylegan.mean_latent(4096).detach()

        # define optimizers
        self.optim_G = Adam(list(self.G.parameters()),
                            lr=self.opt.lr * self.opt.G_lr_ratio,
                            betas=(0.5, 0.999))
        self.optim_F = None  # use data to initialize
        # whether to train the whole D or only the final layer
        if not self.opt.finetune_D_backbone:
            self.optim_D = Adam(list(self.D.final_linear.parameters()),
                                lr=self.opt.lr * self.opt.D_lr_ratio,
                                betas=(0.5, 0.999))
        else:
            self.optim_D = Adam(list(self.D.parameters()),
                                lr=self.opt.lr * self.opt.D_lr_ratio,
                                betas=(0.5, 0.999))
        # define losses
        self.crit_l1 = nn.L1Loss().cuda()
        self.crit_l2 = nn.MSELoss().cuda()
        self.criterionNCE = []
        for _ in self.nce_layers:
            self.criterionNCE.append(PatchNCELoss(self.opt).train().cuda())

        # print and display results
        self.loss_dict = dict()

        # images to save
        self.visual_dict = dict()
        self.visual_names = ['syn_img', 'syn_img_edited']
        self.sample_resize = nn.Upsample(size=self.opt.sample_size)

        print('model_names', self.model_names)
        print('loss_names', self.loss_names)
        print('visual_names', self.visual_names)

    def data_dependent_initialize(self, data):
        self.backward_Ds(data)
        self.backward_Gs()
        self.optim_F = Adam(list(self.F.parameters()), lr=self.opt.lr,
                            betas=(0.5, 0.999))

        # then load the checkpoint
        if self.opt.ckpt is not None:
            ckpt = torch.load(self.opt.ckpt,
                              map_location=lambda storage, loc: storage)
            self.F.load_state_dict(ckpt['F'])
            self.optim_F.load_state_dict(ckpt['optim_F'])

    def load_boundary(self, cond_boundary):
        self.cond_boundary = cond_boundary.cuda().requires_grad_(False)

    @staticmethod
    @torch.no_grad()
    def update_param(net, net_ema, alpha=0.999):
        for param, param_ema in zip(net.parameters(), net_ema.parameters()):
            param_ema.data = param_ema.data * alpha + param.data * (1 - alpha)

    @staticmethod
    def requires_grad(model, flag=True, finetuneD=False):
        if finetuneD:
            for p in model.parameters():
                p.requires_grad = flag
        else:
            for name, p in model.named_parameters():
                if 'final_linear' in name:
                    p.requires_grad = flag

    def generate_img(self, latent, randomize_noise=False, return_feats=False):
        img, _, feats = self.stylegan(
            [latent],
            truncation=self.truncation,
            truncation_latent=self.trunc,
            input_is_latent=True,
            randomize_noise=randomize_noise,
            layers=self.nce_layers
        )
        if return_feats:
            return img, feats
        return img

    @torch.no_grad()
    def generate_latent(self, size=None):
        bs = size if size is not None else self.batch_size
        latent = torch.randn(bs, self.latent_dim).cuda()
        latent = self.stylegan.get_latent(latent)
        return latent

    def run_G(self):
        # first generate latent codes randomly
        with torch.no_grad():
            syn_latent = self.generate_latent()
            # generate the randomly generated images and the feature maps
            syn_img, syn_feats = self.generate_img(syn_latent,
                                                   return_feats=True)

        # edit the latent code to be with the target feature
        syn_latent_edited = self.G(syn_latent,
                                   cond_boundary=self.cond_boundary)

        # generate the edited images and the corresponding feature maps
        syn_img_edited, syn_feats_edited = self.generate_img(syn_latent_edited,
                                                             return_feats=True)
        fake_data = dict(syn_img=syn_img, syn_feats=syn_feats,
                         syn_img_edited=syn_img_edited,
                         syn_feats_edited=syn_feats_edited)
        return fake_data

    @torch.no_grad()
    def calc_rescore(self, predictor, length=10, num=2000):
        rescores = []
        n_batch = num // self.batch_size
        for _ in range(n_batch):
            latent = self.generate_latent()
            img_ori = self.generate_img(latent)
            latent_edited = self.G(latent, length=length,
                                   cond_boundary=self.cond_boundary)
            img_edited = self.generate_img(latent_edited)
            _, score_ori = predictor(nn.Upsample(128)(img_ori))
            _, score_edited = predictor(nn.Upsample(128)(img_edited))
            rescores.append((score_edited - score_ori)[:, 1])
        rescores = torch.cat(rescores, dim=0)
        return rescores.mean(0).item()

    @staticmethod
    def calc_target(pos_rescores, neg_rescores):
        target = sum(pos_rescores) - sum(map(fabs, neg_rescores))
        return target

    def optimize_nce_loss_weights(self, **kwargs):
        if not self.learnable_weights:
            return
        
        length = kwargs.get('length') or 10
        rescore_num = kwargs.get('rescore_num') or 2000
        pos_rescores = [self.calc_rescore(p, length, rescore_num) for p in self.pos_rescore_predictors]
        if self.opt.pos_rescore_minus:
            pos_rescores = [-p for p in pos_rescores]
        neg_rescores = [self.calc_rescore(p, length, rescore_num) for p in self.neg_rescore_predictors]
        
        target = self.calc_target(pos_rescores, neg_rescores)
        loss_weights = self.nce_loss_weights.clone().view(1, -1).cpu().numpy()
        self.weights_list.append(loss_weights)
        self.pos_rescores_list.append(pos_rescores)
        self.neg_rescores_list.append(neg_rescores)
        self.targets_list.append(target)

        new_nce_weights = suggest_weights(self.weights_list, self.targets_list, **kwargs)
        self.optim_weights_infos.update(
            pos_rescores=pos_rescores,
            neg_rescores=neg_rescores,
            old_loss_weights=loss_weights,
            target=target,
            new_loss_weights=new_nce_weights,
        )
        nce_loss_weights = torch.from_numpy(new_nce_weights).cuda()
        self.nce_loss_weights = nce_loss_weights.view(-1)

    def backward_Ds(self, data):
        self.requires_grad(self.G, False)
        self.requires_grad(self.D, True, self.opt.finetune_D_backbone)
        self.requires_grad(self.F, False)

        # forawrd
        real_img = data['img'].cuda()
        fake_data = self.run_G()
        syn_img = fake_data['syn_img']
        syn_img_edited = fake_data['syn_img_edited']

        if self.opt.D_augment:
            real_img = self.augment_pipe(real_img)
            syn_img = self.augment_pipe(syn_img.detach())
            syn_img_edited = self.augment_pipe(syn_img_edited.detach())
        else:
            syn_img = syn_img.detach()
            syn_img_edited = syn_img_edited.detach()

        # D loss
        real_img_pred, _ = self.D(real_img, return_logits=True)
        syn_img_edited_pred = self.D(syn_img_edited)
        d_loss = d_logistic_loss(real_img_pred, syn_img_edited_pred)
        d_loss *= self.opt.lambda_D
        self.loss_dict.update(d=d_loss)

        self.optim_D.zero_grad()
        ds_loss = d_loss
        ds_loss.backward()
        self.optim_D.step()

    def calc_nce_loss(self, _feat_src, _feat_tgt):
        n_layers = len(self.nce_layers)
        feat_src, feat_tgt = [], []
        for (k, _src), (_, _tgt) in zip(_feat_src.items(), _feat_tgt.items()):
            if k not in self.nce_layers:
                continue
            feat_src.append(_src)
            feat_tgt.append(_tgt)
        feat_k_pool, sample_ids = self.F(feat_src, self.opt.num_patches, None)
        feat_q_pool, _ = self.F(feat_tgt, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for i, (f_q, f_k, crit) in enumerate(zip(feat_q_pool, feat_k_pool, self.criterionNCE)):
            loss = crit(f_q, f_k)
            if self.learnable_weights:
                loss *= self.nce_loss_weights[i]
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def backward_Gs(self):
        self.requires_grad(self.G, True)
        self.requires_grad(self.D, False)
        self.requires_grad(self.F, True)

        # forawrd
        fake_data = self.run_G()
        syn_feats = fake_data['syn_feats']
        syn_feats_edited = fake_data['syn_feats_edited']
        syn_img = fake_data['syn_img']
        syn_img_edited = fake_data['syn_img_edited']
        self.visual_dict.update(syn_img=syn_img,
                                syn_img_edited=syn_img_edited)

        # G part of GAN loss
        if self.opt.D_augment:
            syn_img_edited = self.augment_pipe(syn_img_edited)
        syn_img_edited_pred = self.D(syn_img_edited)
        g_loss = g_nonsaturating_loss(syn_img_edited_pred)
        self.loss_dict.update(g=g_loss)

        # patchnce loss
        patchnce_loss = self.calc_nce_loss(syn_feats, syn_feats_edited)
        patchnce_loss *= self.opt.lambda_nce
        self.loss_dict.update(patchnce=patchnce_loss)

        self.optim_G.zero_grad()
        if hasattr(self, 'optim_F') and self.optim_F is not None:
            self.optim_F.zero_grad()
        gs_loss = g_loss + patchnce_loss

        gs_loss.backward()
        self.optim_G.step()
        if hasattr(self, 'optim_F') and self.optim_F is not None:
            self.optim_F.step()

    def optimize_parameters(self, data, n_iter=-1):
        self.backward_Ds(data=data)
        self.backward_Gs()

    def save_networks(self, n_iter):
        ckpt = {
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
            'F': self.F.state_dict(),
            'optim_G': self.optim_G.state_dict(),
            'optim_D': self.optim_D.state_dict(),
            'optim_F': self.optim_F.state_dict(),
        }
        torch.save(ckpt, f'{self.opt.save_dir}/{str(n_iter).zfill(6)}.pt')

    def load_networks(self):
        """load the pretrained checkpoints.

        NOTE:
            1. The feature network F is defined after calling
            `self.data_dependent_initialize`, so the loading of `F` and
            `optim_F` will also be done in `self.data_dependent_initialize`
        """
        print('load model', self.opt.ckpt)
        ckpt = torch.load(self.opt.ckpt,
                          map_location=lambda storage, loc: storage)

        self.G.load_state_dict(ckpt['G'])
        self.D.load_state_dict(ckpt['D'])
        # self.F.load_state_dict(ckpt['F'])
        self.optim_G.load_state_dict(ckpt['optim_G'])
        self.optim_D.load_state_dict(ckpt['optim_D'])
        # self.optim_F.load_state_dict(ckpt['optim_F'])

    def get_visuals(self):
        return self.visual_dict

    def get_losses(self):
        return self.loss_dict

    def get_optim_weights_infos(self):
        return self.optim_weights_infos

    def train(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.train()

    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.eval()

    def test(self, n_iter=-1):
        self.eval()
        imgs = []
        l1s = []
        l2s = []
        cache_filename = 'real_features.npy'
        cache_path = os.path.join(self.opt.img_dir, cache_filename)
        real_features = np.load(cache_path)
        fake_features = []
        cache_filename = 'syn_latents.pt'
        cache_path = os.path.join(self.opt.save_dir, cache_filename)
        if not os.path.exists(cache_path):
            latent = self.generate_latent(self.opt.eval_num)
            torch.save(latent, cache_path)
        else:
            latent = torch.load(cache_path).cuda()

        with torch.no_grad():
            latent_edited = self.G(latent, cond_boundary=self.cond_boundary)
            for l, l_edited in zip(latent, latent_edited):
                img = self.generate_img(l.unsqueeze(0))
                img_edited = self.generate_img(l_edited.unsqueeze(0))
                fake_features.append(self.inception_model(img_edited))
                imgs += [img, img_edited]
                l1s.append(self.crit_l1(img, img_edited).cpu().item())
                l2s.append(self.crit_l2(img, img_edited).cpu().item())
            l1 = np.array(l1s).mean()
            l2 = np.array(l2s).mean()
            fake_features = torch.cat(fake_features,
                                      dim=0).detach().cpu().numpy()
            fid = compute_fid_from_feature(fake_features, real_features)
            fid_filename = 'fid.txt'
            fid_path = os.path.join(self.opt.save_dir, fid_filename)
            with open(fid_path, 'a+') as f:
                msg = f'iter: ({n_iter} / {self.opt.niters}), l1: {l1:.4f}, '
                msg += f'l2: {l2:.4f}, fid: {fid:.4f}\n'
                f.write(msg)

            # the eval images to show
            imgs = torch.cat(imgs, dim=0)

            results = dict(l1=l1, l2=l2, fid=fid, imgs=imgs)

        self.train()

        return results
