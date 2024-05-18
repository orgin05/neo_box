"""Module core/configs provide project base settings"""
import glob
import json
import os
import os.path


class Settings:
    PROJECT_NAME: str = 'Kraken'
    DEVICE = 'neo'
    CONFIG_INFO = {
        'host': '127.0.0.1',
        'port': 8712,
        'channel_count': 9,
        'sample_rate': 1000,
        'buffer_length': 0.5,
        'channel_labels': [
            'CH001',
            'CH002',
            'CH003',
            'CH004',
            'CH005',
            'CH006',
            'CH007',
            'CH008',
            'STIM'
        ],
        'strips': [['CH001', 'CH002', 'CH003', 'CH004'], 
                   ['CH005', 'CH006', 'CH007', 'CH008']],
        'reref': 'bipolar'
    }
    FINGERMODEL_IDS = {
        'rest': 0,
        'cylinder': 1,
        'ball': 2,
        'flex': 3,
        'double': 4,
        'treble': 5,
        'extend': 6,
        'hold': 7,
    }
    FINGERMODEL_IDS_INVERSE = {
        0: 'rest',
        1: 'cylinder',
        2: 'ball',
        3: 'flex',
        4: 'double',
        5: 'treble',
        6: 'extend',
        7: 'hold'
    }
    PROJECT_VERSION: str = '0.0.1'
    DATA_PATH = './data'
    MODEL_PATH = './static/models'


settings = Settings()
