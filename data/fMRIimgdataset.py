import os
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import glob
import PIL.Image
import lmdb
import bdpy

class fMRIImgDataset(Dataset):
    def __init__(self, args, subject='sub-01'):
        self.args = args
        self.subj_data = self.getdata(subject)
        self.sz = len(self.subj_data)

        self.img_transform = transforms.Compose([
                transforms.RandomCrop(227),
                # transforms.ToTensor(),
                transforms.Lambda(lambda image: torch.from_numpy(np.array(image).astype(np.float32))),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return self.sz

    def __getitem__(self, idx):
        fmri, label, img = self.subj_data[idx]
        # print(fmri)
        # print(label)
        # print(img)
        print(img.shape)
        return fmri, self.img_transform(img)
    
    def getdata(self, subj='sub-01'):
        img_size = (248, 248) # For image jittering, we prepare the images to be larger than 227 x 227

        # TODO: Make them params instead of hardcoded values
        # fMRI data :
        fmri_data_table = [
            {'subject': 'sub-01',
            'data_file': './data/fmri/sub-01_perceptionNaturalImageTraining_original_VC.h5',
            'roi_selector': 'ROI_VC = 1',
            'output_dir': './data'},
            {'subject': 'sub-02',
            'data_file': './data/fmri/sub-02_perceptionNaturalImageTraining_original_VC.h5',
            'roi_selector': 'ROI_VC = 1',
            'output_dir': './data'},
            {'subject': 'sub-03',
            'data_file': './data/fmri/sub-03_perceptionNaturalImageTraining_original_VC.h5',
            'roi_selector': 'ROI_VC = 1',
            'output_dir': './data'}
        ]
        # Image data
        image_dir = './data/images/training'
        image_file_pattern = '*.JPEG'

        sbj = fmri_data_table[int(subj[-1])]
        data_arr = []

        # Load fMRI data
        fmri_data_bd = bdpy.BData(sbj['data_file'])
        # Load image files
        images_list = glob.glob(os.path.join(image_dir, image_file_pattern))  # List of image files (full path)
        images_table = {os.path.splitext(os.path.basename(f))[0]: f
                        for f in images_list}                                 # Image label to file path table
        label_table = {os.path.splitext(os.path.basename(f))[0]: i + 1
                    for i, f in enumerate(images_list)}                    # Image label to serial number table

        # Get image labels in the fMRI data
        fmri_labels = fmri_data_bd.get('Label')[:, 1].flatten()

        # Convet image labels in fMRI data from float to file name labes (str)
        fmri_labels = ['n%08d_%d' % (int(('%f' % a).split('.')[0]),
                                    int(('%f' % a).split('.')[1]))
                    for a in fmri_labels]

        # Get sample indexes
        n_sample = fmri_data_bd.dataset.shape[0]

        index_start = 1
        index_end = n_sample
        index_step = 1
        sample_index_list = range(index_start, index_end + 1, index_step)

        # Shuffle the sample indexes
        sample_index_list = np.random.permutation(sample_index_list)

        # Save the sample indexes
        save_name = subj + '_sample_index_list.txt'
        np.savetxt(os.path.join(sbj['output_dir'], save_name), sample_index_list, fmt='%d')

        # Get fMRI data in the ROI
        fmri_data = fmri_data_bd.select(sbj['roi_selector'])

        # Normalize fMRI data
        fmri_data_mean = np.mean(fmri_data, axis=0)
        fmri_data_std = np.std(fmri_data, axis=0)

        fmri_data = (fmri_data - fmri_data_mean) / fmri_data_std

        for j0, sample_index in np.ndenumerate(sample_index_list):
            sample_label = fmri_labels[sample_index - 1]  # Sample label (file name)
            sample_label_num = label_table[sample_label]  # Sample label (serial number)

            # fMRI data in the sample
            sample_data = fmri_data[sample_index - 1, :]
            sample_data = np.float64(sample_data)  
            sample_data = np.reshape(sample_data, (sample_data.size, 1, 1))  # Num voxel x 1 x 1

            # Load images
            image_file = images_table[sample_label]
            img = PIL.Image.open(image_file)
            img = np.array(img.resize(img_size, PIL.Image.BILINEAR))

            if img.ndim == 2: # Monochrome --> RGB
                img_rgb = np.zeros((img_size[0], img_size[1], 3), dtype=img.dtype)
                img_rgb[:, :, 0] = img
                img_rgb[:, :, 1] = img
                img_rgb[:, :, 2] = img
                img = img_rgb

            img = img.transpose(2, 0, 1)  # h x w x c --> c x h x w
            img = img[::-1]               # RGB --> BGR

            data_arr.append([])
            data_arr[-1].append(sample_data)
            data_arr[-1].append(sample_label_num)
            data_arr[-1].append(img)

        print("Done")
        return data_arr