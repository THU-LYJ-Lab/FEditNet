from typing import Any
from copy import deepcopy


class LossDict(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value
    
    def __str__(self):
        try:
            msg = ', '.join(
                [f'{k}: {float(v):.4f}' for k, v in self.items()])
        except TypeError as err:
            print(f'{err} in {self}')
            raise
        return msg

    def __add__(self, other):
        res = LossDict()
        if type(self) != type(other):
            raise TypeError(f'Type of both should be LossDict, however '
                            f'({type(self)}, {type(other)}) received!')
        for key in self:
            if key not in other:
                raise KeyError
            try:
                res[key] = self[key] + other[key]
            except TypeError:
                raise TypeError(f'Error at key `{key}`.')
        return res
    
    def __mul__(self, scalar):
        res = LossDict()
        for key in self:
            try:
                res[key] = self[key] * scalar
            except TypeError:
                raise TypeError(f'Error at key `{key}`.')
        return res


class LossDicts():
    def __init__(self, allowed_keys: list = []):
        self.loss_dicts = dict()
        self.allowed_keys = allowed_keys

    def add_dict(self, name, loss_dict: LossDict):
        if set(self.allowed_keys) != set(loss_dict.keys()):
            raise ValueError(f'Invalid keys of loss dict: '
                             f'{list(loss_dict.keys())}! Allowed keys: '
                             f'{self.allowed_keys}.')
        if name in self.loss_dicts:
            raise KeyError(f'LossDict `{name}`` has already exists!')

        self.loss_dicts[name] = loss_dict
    
    def __str__(self):
        msg = '\n'.join(
            [f'{name:<8s} {loss_dict}' for name, loss_dict in self.loss_dicts.items()])
        return msg
    
    def __add__(self, other):
        if type(self) != type(other):
            raise TypeError(f'Type of both should be LossDicts, however '
                            f'({type(self)}, {type(other)}) received!')
        if set(self.allowed_keys) != set(other.allowed_keys):
            raise ValueError(f'Two loss dicts should have the same allowed keys, '
                             f'but ({self.allowed_keys}, {other.allowed_keys})'
                             f' received.')
        res = deepcopy(self)
        for name in other.loss_dicts:
            if name not in self.loss_dicts:
                res.add_dict(name, other.loss_dicts[name])
            else:
                res.loss_dicts[name] = self.loss_dicts[name] + other.loss_dicts[name]
        return res
    
    def __mul__(self, scalar):
        res = LossDicts(self.allowed_keys)
        for name, loss_dict in self.loss_dicts.items():
            res.add_dict(name, loss_dict * scalar)
        return res

    def sorted(self, reverse=False):
        res = deepcopy(self)
        for name, loss_dict in self.loss_dicts.items():
            res.loss_dicts[name] = sorted(loss_dict.items(), key=lambda x: x[1], reverse=reverse)
        return res


def _ratio(a: LossDict, b: LossDict) -> LossDict:
    res = LossDict()
    if set(a.keys()) != set(b.keys()):
        raise ValueError(f'Requires the same keys!')
    for (k, v1), v2 in zip(a.items(), b.values()):
        res[k] = float(v1) / float(v2)
    return res


def ratio(a: LossDicts, b: LossDicts) -> LossDicts:
    if set(a.allowed_keys) != set(b.allowed_keys):
        raise ValueError(f'Two loss dicts should have the same allowed keys, '
                         f'but ({a.allowed_keys}, {b.allowed_keys})'
                         f' received.')
    if set(a.loss_dicts.keys()) != set(b.loss_dicts.keys()):
        raise ValueError(f'Two loss dicts should have the same loss dict keys, '
                         f'but ({set(a.loss_dicts.keys())}, '
                         f'{set(b.loss_dicts.keys())}) received.')
    res = LossDicts(a.allowed_keys)
    for name in a.loss_dicts:
        res.add_dict(name, _ratio(a.loss_dicts[name], b.loss_dicts[name]))
    return res
