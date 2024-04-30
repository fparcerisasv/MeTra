import os
import sys
sys.path.insert(0, os.getcwd())

#from classification.datasets import MIMICDataset, UKADataset, load_cxr_ehr, my_collate
from classification.datasets.mimic_lab import load_cxr_ehr, my_collate


def get_dataset(cfg):
    collate_fn = None
    if cfg.dataset.name == 'MIMICLab':
        train_dataset, validation_dataset, test_dataset = load_cxr_ehr(cfg=cfg)
        collate_fn = my_collate

    return train_dataset, validation_dataset, test_dataset, collate_fn
