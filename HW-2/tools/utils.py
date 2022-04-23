# -*- coding: utf-8 -*-
# @Time : 2022/4/24 14:52
# @Author : Mengtian Zhang
# @Version : v-dev-0.0
# @Function 

"""Summary.

Description.----------------------------------------------------------------
----------------------------------------------------------------------------
----------------------------------------------------------------------------

Example:

"""

import os
import time


def get_timestamp(file_name=None):
    localtime = time.localtime(time.time())
    date_time = "{}_{}_{}_{}_{}_{}".format(localtime.tm_year, localtime.tm_mon, localtime.tm_mday, localtime.tm_hour,
                                           localtime.tm_min, localtime.tm_sec)
    if file_name:
        return file_name + '_' + date_time
    return date_time


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def seed_everything(seed: int = 0):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
