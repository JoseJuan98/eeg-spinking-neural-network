# -*- coding: utf-8 -*-
"""Preprocessing modules."""


import os
import pathlib

import numpy as np
import torch
import scipy.io
import scipy.signal
import picard
import sklearn.preprocessing
import matplotlib.pyplot as plt
import torch


from common.config import Config

def get_dataset(file_name: str | pathlib.Path):

    dataset = torch.load(f=Config.data_dir / file_name, weights_only=True)

    data = dataset['data']
    labels = dataset['labels']
    subjects = dataset['subjects']