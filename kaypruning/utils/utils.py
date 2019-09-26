from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from pprint import pformat

import numpy as np
import pydash as ps
import tensorflow as tf
import torch


class DotDict(dict):
    """
    Dictionary to access attributes
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# -------------------- Seed:Global -------------------- #


def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------- Info:Describe -------------------- #


def get_cls_name(obj, lower=False):
    r"""
    Get the class name of an object
    """

    class_name = obj.__class__.__name__
    if lower:
        class_name = class_name.lower()
    return class_name


def get_cls_attr(obj):
    r"""
    Get the class attr of an object as dict
    """
    attr_dict = {}
    for k, v in obj.__dict__.items():
        if hasattr(v, '__dict__'):
            val = str(v)
        else:
            val = v
        attr_dict[k] = val
    return attr_dict


def describe(cls):
    desc_list = [f'{get_cls_name(cls)}:']
    for k, v in get_cls_attr(cls).items():
        if k.startswith('_'):
            continue
        elif ps.is_dict(v) or ps.is_dict(ps.head(v)):
            desc_v = pformat(v)
        else:
            desc_v = v
        desc_list.append(f'- {k} = {desc_v}')  # \t| type -> {type(desc_v)}')
    desc = '\n'.join(desc_list) + '\n\n'
    return desc

# -------------------- Parser:Create -------------------- #
