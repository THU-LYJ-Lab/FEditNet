import torch
import os
import numpy as np
import cv2
import random
import scipy.linalg


def compute_fid_from_feature(fake_features, real_features):
    fake_mean = np.mean(fake_features, axis=0)
    fake_cov = np.cov(fake_features, rowvar=False)
    real_mean = np.mean(real_features, axis=0)
    real_cov = np.cov(real_features, rowvar=False)

    fid = np.square(fake_mean - real_mean).sum()
    temp = scipy.linalg.sqrtm(np.dot(fake_cov, real_cov))
    fid += np.real(np.trace(fake_cov + real_cov - 2 * temp))
    return float(fid)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def ortho(tensor, ortho_tensor_list):
    for t in ortho_tensor_list:
        tensor = tensor - (tensor*t).sum() * t
    return tensor / tensor.norm()


def GramSchmidt(tensors):
    tensors = torch.cat([x.unsqueeze(0) / x.norm() for x in tensors], 0)
    if len(tensors) < 2:
        return tensors
    ortho_tensor_list = []
    for tensor in tensors:
        if len(ortho_tensor_list) < 1:
            ortho_tensor_list.append(tensor)
        else:
            ortho_tensor_list.append(ortho(tensor, ortho_tensor_list).unsqueeze(0))
    return torch.cat(ortho_tensor_list, 0)


def my_make_grid(image_dict):
    columns = []
    for n, v in image_dict.items():
        column = torch.cat([v[i] for i in range(v.shape[0])], dim=1)
        columns.append(column)
    return torch.cat(columns, dim=2) / 2 + 0.5


if __name__ == '__main__':
    set_seed(0)
    print(torch.randn(10))