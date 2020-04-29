import os
import torch
import numpy as np
from torch.utils.data import Dataset
import pickle

class fMRIImgDataset(Dataset):
    def __init__(self, args, subject='sub-01'):
        self.args = args
        all_data = pickle.load(open(os.path.join(args.dataroot, 'fMRIImgData-' + subject + '.p'), 'rb'))
        self.subj_data = all_data[subject]
        self.sz = len(self.subj_data)

    def __len__(self):
        return self.sz

    def __getitem__(self, idx):
        return self.subj_data[idx]
