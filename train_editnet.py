"""The main file for training."""
import os
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from editnet import EditNet
from data import build_dataloader, sample_data
from utils.parser import parse_args
from utils.utils import set_seed, my_make_grid


def train(opt):
    # basic settings
    set_seed(opt.seed)
    torch.autograd.set_detect_anomaly(True)
    start_iter = 1
    # flag for initialization of patch nce loss
    model_init_flag = False

    # logger of TensorBoard
    logger = SummaryWriter(opt.log_dir)

    # build the model and load the checkpoint
    editnet = EditNet(opt)
    if opt.ckpt is not None:
        editnet.load_networks()
        try:
            ckpt_name = os.path.basename(opt.ckpt)
            start_iter = 1 + int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

    # use to save data for tensorboard
    sum_loss = {k: 0. for k in editnet.loss_names}

    # build the dataset
    dataloader = build_dataloader(opt, opt.seed)
    dataloader = sample_data(dataloader)

    # load conditional boundary
    if opt.boundary_dir is not None:
        boundary = torch.load(opt.boundary_dir)
        editnet.load_boundary(boundary)

    pbar = tqdm(range(opt.niters), initial=start_iter,
                dynamic_ncols=True, smoothing=0.01)
    # the main iteration
    for i in pbar:
        if i + start_iter > opt.niters:
            print('training completed')
            break
        data = next(dataloader)

        # initialize F for patch nce loss
        if not model_init_flag:
            model_init_flag = True
            editnet.data_dependent_initialize(data)

        # the main part of training
        editnet.optimize_parameters(data, i + start_iter)
        # update losses for tensorboard
        loss_dict = editnet.get_losses()
        # print to console
        msg = ''
        msg += ', '.join(
            [f'{n}: {k.cpu().item():.4f}' for n, k in loss_dict.items()])
        pbar.set_description(msg)

        # record the losses
        for n, v in loss_dict.items():
            sum_loss[n] += v

        if (i + start_iter) % opt.optim_freq == 0:
            editnet.optimize_nce_loss_weights(**opt.optim_kwargs)
            optim_infos = editnet.get_optim_weights_infos()
            info = f'iter: {(i + start_iter)}/{opt.niters}, optimize nce weights\n'
            with open(f'{opt.save_dir}/learnable_loss_weights.txt', 'a+') as f:
                for k, v in optim_infos.items():
                    info += f'{k:>16s}: {v}\n'
                info += '\n'
                f.write(info)
                print(info)

        # write the losses to the tensorboard
        if (i + start_iter) % opt.print_freq == 0:
            # write to tensorboard
            for name, loss in sum_loss.items():
                logger.add_scalar(f'Loss/loss_{name}',
                                  loss / opt.print_freq,
                                  i + start_iter)
                sum_loss[name] = 0.

        if (i + start_iter) % opt.display_freq == 0:
            image_dict = editnet.get_visuals()
            # save the images to the disk
            save_image(
                my_make_grid(image_dict),
                f'{opt.sample_dir}/{i + start_iter}.jpg',
                nrow=opt.batch_size,
                normalize=True,
                range=(0, 1))
            # write to tensorboard
            logger.add_image(f'{i + start_iter}/edited',
                             my_make_grid(image_dict),
                             i + start_iter)

        if (i + start_iter) % opt.eval_freq == 0:
            results = editnet.test(i + start_iter)
            imgs = results['imgs']
            l1 = results['l1']
            l2 = results['l2']
            fid = results['fid']
            logger.add_scalar(f'Metric/l1', l1, i + start_iter)
            logger.add_scalar(f'Metric/l2', l2, i + start_iter)
            logger.add_scalar(f'Metric/fid', fid, i + start_iter)
            save_image(
                imgs,
                f'{opt.sample_dir}/{i + start_iter}_eval.jpg',
                nrow=opt.eval_nrow,
                normalize=True,
                range=(-1, 1))

        if (i + start_iter) % opt.save_freq == 0:
            editnet.save_networks(i+start_iter)
            print(f'saving the model (epoch {i + start_iter} / {opt.niters})')

    logger.close()


if __name__ == '__main__':
    args = parse_args()
    train(args)
