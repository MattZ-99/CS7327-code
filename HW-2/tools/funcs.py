# -*- coding: utf-8 -*-
# @Time : 2022/4/24 15:08
# @Author : Mengtian Zhang
# @Version : v-dev-0.0
# @Function 

"""Summary.

Description.----------------------------------------------------------------
----------------------------------------------------------------------------
----------------------------------------------------------------------------

Example:

"""

import argparse


def get_args(*args, **kwargs):
    parser = argparse.ArgumentParser(description='CS7327-HW2-SEED-Transfer-Learning.')
    parser.add_argument('--gpu', default=0, type=int, help='GPU number.')
    parser.add_argument('--epochs', default=50, type=int, help='Training epochs.')
    parser.add_argument('--batch-size', default=128, type=int, help='Batch size.')
    parser.add_argument('--seed', default=1234, type=int, help='Seed for everything.')
    parser.add_argument('--alpha-cft', '-acft', default=1., type=float, help='Coefficient of alpha.')
    parser.add_argument('--output-dir', default='./Outputs', type=str, help='Root output path.')
    args = parser.parse_args(*args, **kwargs)

    return args
