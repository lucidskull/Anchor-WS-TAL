import os
import pprint
import random

import torch
import numpy as np

from ..config import cfg
from ..models.spn import TemporalSegmentProposalNetwork
from ..models.soi import TemporalSoINetwork
from ..models.models import ActionProposalGenerator

def init_spn_model():
    spn = TemporalSegmentProposalNetwork()

    return spn

def init_soi_model():
    soi = TemporalSoINetwork(cfg.SPN_OUT, cfg.SOI_SIZE)

    return soi

def init_model(spn, soi):
    tscnn = ActionProposalGenerator(spn, soi)

    return tscnn

def load_model():
    pass

def makedirs(dir):
    if not os.path.exists(dir): 
        os.makedirs(dir)

def set_path(config):
    if config.MODE == 'train':
        config.EXP_NAME = 'experiments/{cfg.MODE}/{cfg.NUM_ITERS}'.format(cfg=config)
        config.MODEL_PATH = os.path.join(config.EXP_NAME, 'model')
        config.LOG_PATH = os.path.join(config.EXP_NAME, 'log')

        if not os.path.exists(config.MODEL_PATH): 
            os.makedirs(config.MODEL_PATH)
        if not os.path.exists(config.LOG_PATH): 
            os.makedirs(config.LOG_PATH)

    elif config.MODE == 'test':
        config.EXP_NAME = 'experiments/{cfg.MODE}'.format(cfg=config)

    config.OUTPUT_PATH = os.path.join(config.EXP_NAME, 'output')
    makedirs(config.OUTPUT_PATH)

def save_config(config):
    file_path = os.path.join(config.OUTPUT_PATH, "config.txt")
    fo = open(file_path, "w")
    fo.write("Configurtaions:\n")
    fo.write(pprint.pformat(config))
    fo.close()

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False