import numpy as np
from numpy import random
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor


class Acquisition(object):
    def __init__(self, kappa, xi, acq_type='ei'):
        self.kappa = kappa
        self.xi = xi
        self.acq_type = acq_type

    def calc_acq(self, x, gp, y_max):
        if self.acq_type == 'ucb':
            return self._ucb(x, gp, self.kappa)
        if self.acq_type == 'ei':
            return self._ei(x, gp, y_max, self.kappa)
        if self.acq_type == 'poi':
            return self._poi(x, gp, y_max, self.xi)

    @staticmethod
    def _ucb(x, gp, kappa):
        mean, std = gp.predict(x, return_std=True)
        return mean + kappa * std

    @staticmethod
    def _ei(x, gp, y_max, xi):
        mean, std = gp.predict(x, return_std=True)
        a = (mean - y_max - xi)
        z = a / std
        return a * norm.cdf(z) + std * norm.pdf(z)

    @staticmethod
    def _poi(x, gp, y_max, xi):
        mean, std = gp.predict(x, return_std=True)
        z = (mean - y_max - xi)/std
        return norm.cdf(z)


def generate_weights(random_state, bounds, num, normal=False):
    _weights = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                    size=(num, bounds.shape[0]))
    if not normal:
        return _weights
    return _weights / _weights.sum(1).reshape(-1, 1) * bounds.shape[0]
    


def suggest_weights(wts_list, tgt_list, **kwargs):
    # TODO: how to set bounds?
    n_layers = wts_list[0].shape[1]
    bounds = np.concatenate((np.zeros((n_layers, 1)), np.ones((n_layers, 1))), axis=1)
    bounds = bounds * 2
    seed = kwargs.get('seed') or 0
    normal = kwargs.get('normal') or False
    random_state = np.random.RandomState(seed)

    kappa = kwargs.get('kappa') or 2.576
    xi = kwargs.get('xi') or 0.0
    acq_type = kwargs.get('acq_type') or 'ei'
    acq_util = Acquisition(kappa, xi, acq_type)

    nce_weights = np.concatenate(wts_list, axis=0)
    targets = np.array(tgt_list)

    const_kernel = kwargs.get('const_kernel') or 1.0
    const_kernel_range = kwargs.get('const_kernel_range') or (1e-3, 1e3)
    rbf_kernel_scale = kwargs.get('rbf_kernel_scale') or 10
    rbf_kernel_range = kwargs.get('rbf_kernel_range') or (0.5, 2)
    n_restarts_optimizer = kwargs.get('n_restarts_optimizer') or 9
    kernel = C(const_kernel, const_kernel_range) * RBF(rbf_kernel_scale, rbf_kernel_range)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer)
    gp.fit(nce_weights, targets)

    # TODO: search, use target to calculate acquisition
    n_warmup = kwargs.get('n_warmup') or 10000
    n_iter = kwargs.get('n_iter') or 10
    x_trials = generate_weights(random_state, bounds, n_warmup, normal)
    y_max = targets.max()
    ys = acq_util.calc_acq(x_trials, gp, y_max)
    x_max = x_trials[ys.argmax()]
    max_acq = ys.max()

    x_seeds = generate_weights(random_state, bounds, n_iter, normal)
    for x_try in x_seeds:
        res = minimize(lambda x: -acq_util.calc_acq(x.reshape(1, -1), gp=gp, y_max=y_max),
                       x_try.reshape(1, -1),
                       bounds=bounds,
                       method="L-BFGS-B")
        if not res.success:
            continue
        if max_acq is None or -res.fun[0] >= max_acq:
            x_max = res.x
            max_acq = -res.fun[0]

    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    new_weights = np.clip(x_max, bounds[:, 0], bounds[:, 1])
    if not normal:
        return new_weights
    return new_weights / new_weights.sum() * bounds.shape[0]
    