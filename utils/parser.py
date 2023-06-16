import os
import argparse
from math import sqrt, log2


class Parser(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        # basic settings
        parser.add_argument('--seed', type=int, default=0, help='random seed')
        parser.add_argument('--name', type=str, default='demo', help='experiment name')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gou ids for training')
        
        # model settings
        parser.add_argument('--prob_crit', type=int, default=0.2, help='whether to use fix length for boundary')
        parser.add_argument('--fix_len', type=int, default=5, help='whether to use fix length for boundary')
        parser.add_argument('--fake_queue_size', type=int, default=1000, help='queue size for fake head')
        parser.add_argument('--hidden_channels', type=int, default=512, help='hidden channel for fake head')
        
        # dataset settings
        parser.add_argument('--img_dir', type=str, default='./datasets/celeba_data/train_Old', help='path to training data')
        parser.add_argument('--latent_dir', type=str, default='./datasets/celeba_data/train_Old', help='path to training data')
        parser.add_argument('--batch_size', type=int, default=4, help='batch size for one single iteration')
        parser.add_argument('--num_shot', type=int, default=None, help='how many data can be seen')
        
        # StyleGAN settings
        parser.add_argument('--latent_dim', type=int, default=512, help='dimension of latent space for StyleGAN2')
        parser.add_argument('--img_size', type=int, default=512, help='image size for StyleGAN2')
        parser.add_argument('--n_mlp', type=int, default=8, help='how many mlps in StyleGAN2')
        parser.add_argument('--truncation', type=float, default=0.7, help='parameter for styleGAN2')
        # parser.add_argument('--stylegan_ckpt', type=str, default='./stylegan_checkpoints/celeba/740000.pt', help='path to checkpoint of pretrained StyleGAN2')
        parser.add_argument('--stylegan_ckpt', type=str, default='./stylegan_checkpoints/ffhq/390000.pt', help='path to checkpoint of pretrained StyleGAN2')
        # parser.add_argument('--stylegan_ckpt', type=str, default='./stylegan_checkpoints/afhq/170000.pt', help='path to checkpoint of pretrained StyleGAN2')
        # parser.add_argument('--stylegan_ckpt', type=str, default='./stylegan_checkpoints/LSUN_tower/220000.pt', help='path to checkpoint of pretrained StyleGAN2')
        # parser.add_argument('--stylegan_ckpt', type=str, default='./stylegan_checkpoints/LSUN_church/270000.pt', help='path to checkpoint of pretrained StyleGAN2')
        
        parser.add_argument('--D_augment', action='store_true', help='whether to add augmentations before D')
        parser.add_argument('--finetune_D_backbone', action='store_true', help='whether to train only the final layer of D')

        # patch nce settings
        parser.add_argument('--netF_nc', type=int, default=256, help='parameter for netF in CUT')
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--nce_layers', type=str, default='4,8,11,12,13,14', help='compute NCE loss on which layers of StyleGAN2')
        # parser.add_argument('--nce_layers', type=str, default='4,7,9,10,11,12', help='compute NCE loss on which layers of StyleGAN2')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch', action='store_true')

        # learnable nce weights
        parser.add_argument('--learnable_weights', action='store_true')
        parser.add_argument('--normal', action='store_true')
        parser.add_argument('--rescore_length', type=float, default=10)
        parser.add_argument('--rescore_num', type=int, default=2000)
        parser.add_argument('--optim_freq', type=int, default=1000)
        parser.add_argument('--kappa', type=float, default=2.576)
        parser.add_argument('--xi', type=float, default=0.0)
        parser.add_argument('--acq_type', type=str, default='ei')
        parser.add_argument('--const_kernel', type=float, default=1.0)
        parser.add_argument('--const_kernel_range', type=str, default='1e-3,1e3')
        parser.add_argument('--rbf_kernel_scale', type=float, default=10)
        parser.add_argument('--rbf_kernel_range', type=str, default='0.5,2')
        parser.add_argument('--n_restarts_optimizer', type=int, default=9)
        parser.add_argument('--n_warmup', type=int, default=10_000)
        parser.add_argument('--n_iter', type=int, default=10)
        parser.add_argument('--pos_rescore_ckpts', type=str, default='./checkpoints/resnet_predictor/Smiling_10.pt')
        parser.add_argument('--neg_rescore_ckpts', type=str, default='./checkpoints/resnet_predictor/Male_10.pt')
        parser.add_argument('--pos_rescore_minus', action='store_true')
        
        # loss weights
        parser.add_argument('--lambda_D', type=float, default=1., help='weight for D loss')
        parser.add_argument('--lambda_nce', type=float, default=3., help='weight for patchnce loss')
        
        # training settings
        parser.add_argument('--niters', type=int, default=30000, help='how many iterations to train')
        parser.add_argument('--lr', type=float, default=0.002, help='learning rate for Adam')
        parser.add_argument('--G_lr_ratio', type=float, default=1., help='learning rate ratio for G')
        parser.add_argument('--D_lr_ratio', type=float, default=0.1, help='learning rate ratio for D')

        # saving settings
        parser.add_argument('--boundary_dir', type=str, default=None, help='path to conditional boundary')
        parser.add_argument('--ckpt', type=str, default=None, help='pretrained model to continue training')
        parser.add_argument('--log_dir', type=str, default='./logs/ffhq', help='path to save log file and tensorboard')
        parser.add_argument('--save_dir', type=str, default='./checkpoints/ffhq', help='path to save checkpoints')
        parser.add_argument('--sample_dir', type=str, default='./samples/ffhq', help='path to save sample images')
        parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results')
        parser.add_argument('--print_freq', type=int, default=50, help='frequency of showing training losses')
        parser.add_argument('--save_freq', type=int, default=1000, help='frequency of saving checkpoints')
        parser.add_argument('--sample_size', type=int, default=256, help='sample size')

        # eval settings
        parser.add_argument('--inception_ckpt', type=str, default='./checkpoints/pt_inception-2015-12-05-6726825d.pth')
        parser.add_argument('--eval_freq', type=int, default=1000, help='frequency of evaluating the models')
        parser.add_argument('--eval_num', type=int, default=64, help='number for one single evaluation')

        self.parser = parser

    def print_opt(self, parser, save_flag=True):
        opt = parser.parse_args()
        opt.save_dir = os.path.join(opt.save_dir, opt.name)
        if not os.path.exists(opt.save_dir):
            os.makedirs(opt.save_dir)
        opt.sample_dir = os.path.join(opt.sample_dir, opt.name)
        if not os.path.exists(opt.sample_dir):
            os.makedirs(opt.sample_dir)
        if not os.path.exists(opt.log_dir):
            os.makedirs(opt.log_dir)

        # nce layers
        max_layer = 2 * int(log2(opt.img_size) - 2) + 1
        if not opt.learnable_weights:
            opt.nce_layers = [int(i) for i in opt.nce_layers.split(',')]
            assert max(opt.nce_layers) <= max_layer
        else:
            opt.nce_layers = list(range(max_layer + 1))
            # rescore ckpts
            if opt.pos_rescore_ckpts is not None:
                opt.pos_rescore_ckpts = opt.pos_rescore_ckpts.split(',')
            else:
                opt.pos_rescore_ckpts = []
            if opt.neg_rescore_ckpts is not None:
                opt.neg_rescore_ckpts = opt.neg_rescore_ckpts.split(',')
            else:
                opt.neg_rescore_ckpts = []
            assert len(opt.pos_rescore_ckpts) + len(opt.neg_rescore_ckpts) > 0
        
        opt.const_kernel_range = tuple(map(float, opt.const_kernel_range.split(',')))
        opt.rbf_kernel_range = tuple(map(float, opt.rbf_kernel_range.split(',')))
        
        message = ''
        message += '----------------- Options ---------------\n'
        opt.log_dir = os.path.join(opt.log_dir, opt.name)
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        msg = '\n----------------- Model Settings ---------------\n'
        msg += f'use fixed length of {opt.fix_len}\n\n'
        if not opt.finetune_D_backbone:
            msg += f'only train the final layer of D while keeping the backbone frozen\n\n'
        else:
            msg += f'train the whole D\n\n'
        if not opt.learnable_weights:
            msg += f'use fixed layers of nce loss with {opt.nce_layers}\n'
        else:
            msg += f'use learnable nce loss weights with all layers {opt.nce_layers}\n'
            msg += f'use positive rescore ckpts:\n\t'
            msg += f'\n\t'.join(opt.pos_rescore_ckpts)
            msg += f'\nuse negative rescore ckpts:\n\t'
            msg += f'\n\t'.join(opt.neg_rescore_ckpts)
        msg += '\n----------------- End -------------------\n'
        print(message)
        print(msg)
        # save to the disk
        if save_flag:
            name = os.path.join(opt.save_dir, 'opt.txt')
            with open(name, 'wt') as opt_file:
                opt_file.write(message)
                opt_file.write(msg)
                opt_file.write('\n')
        return opt

    def parse_args(self):
        opt = self.print_opt(self.parser)
        opt.gpu_ids = list(map(int, opt.gpu_ids.split(',')))

        # options for building models
        opt.G_kwargs = dict(
            latent_dim=opt.latent_dim,
            fix_len=opt.fix_len,
            init_boundary=None,
        )
        opt.F_kwargs = dict(
            use_mlp=True,
            nc=opt.netF_nc,
            gpu_ids=opt.gpu_ids,
        )
        opt.stylegan_kwargs = dict(
            size=opt.img_size,
            style_dim=opt.latent_dim,
            n_mlp=opt.n_mlp,
        )
        opt.D_kwargs = dict(
            size=opt.img_size,
        )
        opt.Gcl_kwargs = dict(
            hidden_channels=opt.hidden_channels,
            queue_size=opt.fake_queue_size
        )
        opt.C_kwargs = dict(
            size=opt.img_size,
        )
        opt.optim_kwargs = dict(
            normal=opt.normal,
            length=opt.rescore_length,
            rescore_num=opt.rescore_num,
            seed=opt.seed,
            kappa=opt.kappa,
            xi=opt.xi,
            acq_type=opt.acq_type,
            const_kernel=opt.const_kernel,
            const_kernel_range=opt.const_kernel_range,
            rbf_kernel_scale=opt.rbf_kernel_scale,
            rbf_kernel_range=opt.rbf_kernel_range,
            n_restarts_optimizer=opt.n_restarts_optimizer,
            n_warmup=opt.n_warmup,
            n_iter=opt.n_iter,
        )

        # options for display eval results
        eval_nrow = int(sqrt(opt.eval_num))
        if eval_nrow * eval_nrow != opt.eval_num:
            raise ValueError(f'The number of eval images `eval_num` should '
                             f'be a square number, however, {opt.eval_num} '
                             f'received!')
        opt.eval_nrow = eval_nrow

        # options for augmentation
        '''
        opt.aug_kwargs = dict(xflip=1, rotate90=0, xint=1, scale=1,
                              rotate=1, aniso=1, xfrac=0,
                              brightness=1, contrast=1, lumaflip=0,
                              hue=0, saturation=1, noise=1,
                              rotate_max=0.17)
        '''
        opt.aug_kwargs = dict(xflip=1, rotate90=0, xint=1, scale=1,
                              rotate=1, aniso=1, xfrac=0, brightness=0,
                              contrast=0, lumaflip=0, hue=0,
                              saturation=0, noise=0)

        # return
        self.opt = opt
        return self.opt


def parse_args():
    parser = Parser()
    return parser.parse_args()