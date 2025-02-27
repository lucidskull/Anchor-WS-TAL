import os

import numpy as np

class Config():
    def __init__(self):

        # SETTINGS
        self.GPU_ID = '0'
        self.SEED = 0
        self.BATCH_SIZE = 8
        self.NUM_WORKERS = 8

        # DATA
        self.DATA_PATH = './data/THUMOS14'
        self.FEATS_FPS = 25
        self.NUM_SEGMENTS = 500

        # METADATA
        self.feature_dim = 2048
        self.receptive_fields = [1,3,5,8]
        self.NUM_CLASSES = 20
        self.classes = {
            'BaseballPitch': 0,
            'BasketballDunk': 1,
            'Billiards': 2, 
            'CleanAndJerk': 3,
            'CliffDiving': 4,
            'CricketBowling': 5, 
            'CricketShot': 6,
            'Diving': 7,
            'FrisbeeCatch': 8, 
            'GolfSwing': 9,
            'HammerThrow': 10,
            'HighJump': 11, 
            'JavelinThrow': 12,
            'LongJump': 13,
            'PoleVault': 14, 
            'Shotput': 15,
            'SoccerPenalty': 16,
            'TennisSwing': 17, 
            'ThrowDiscus': 18,
            'VolleyballSpiking': 19}

        # TRAIN
        self.NUM_ITERS = 6000
        self.TEST_FREQ = 100
        self.LR_SPN = '[0.00001]*6000'
        self.LR_SOI = '[0.0001]*6000'

        # EVAL
        self.CLASS_THRESH = 0.2
        self.NMS_THRESH = 0.6
        self.CAS_THRESH = np.arange(0.0, 0.25, 0.025)
        self.ANESS_THRESH = np.arange(0.1, 0.925, 0.025)
        self.TIOU_THRESH = np.linspace(0.1, 0.7, 7)
        self.UP_SCALE = 24
        self.GT_PATH = os.path.join(self.DATA_PATH, 'gt.json')

        # LOG
        self.PRINT_FREQ = 10

        # OUTPUT
        self.EXP_NAME = ""

        # SPN
        self.SPN_OUT = 2048

        # SOI
        self.SOI_SIZE = 5

cfg = Config()