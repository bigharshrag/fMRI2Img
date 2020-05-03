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

FMRI_DIMENSIONS = 18000

class fMRIImgDataset(Dataset):
    def __init__(self, args, subject='sub-01', conv=False):
        self.args = args
        self.subj_data = self.getdata(subject)
        self.sz = len(self.subj_data)

        if conv == False:
            self.img_transform = transforms.Compose([
                    transforms.Resize(280),
                    transforms.RandomCrop(256),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
        else:
            self.img_transform = transforms.Compose([
                    transforms.Resize(280),
                    transforms.RandomCrop(256),
                    transforms.ColorJitter(brightness=0.3, contrast=0.3),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

    def __len__(self):
        return self.sz

    def __getitem__(self, idx):
        fmri, img, class_label, class_idx, label = self.subj_data[idx]
        return fmri, self.img_transform(img), class_label, class_idx
    
    def getdata(self, subj='sub-01'):
        # img_size = (248, 248) # For image jittering, we prepare the images to be larger than 227 x 227

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
        image_dir = './data/deeprecon_stimuli_20190212/stimuli/natural-image_training'
        image_file_pattern = '*.JPEG'

        sbj = fmri_data_table[int(subj[-1]) - 1]
        data_arr = []

        # Load fMRI data
        fmri_data_bd = bdpy.BData(sbj['data_file'])

        def getClassName(img):
            return img[:img.find('_')]

        # Load image files
        images_list = glob.glob(os.path.join(image_dir, image_file_pattern))  # List of image files (full path)
        image_names = [os.path.splitext(os.path.basename(f))[0] for f in images_list]
        images_table = {os.path.splitext(os.path.basename(f))[0]: f for f in images_list}                                 # Image label to file path table
        label_table = {n: i + 1 for i, n in enumerate(image_names)}  
        class_list = list(set([n[:n.find('_')] for n in image_names]))
        class_table = {}
        class_counts = {}
        for img in image_names:
            img_class = getClassName(img)
            class_idx = class_list.index(img_class)
            if img_class not in class_counts:
                class_table[img] = (class_idx, 0)
                class_counts[img_class] = 1
            else:
                class_table[img] = (class_idx, class_counts[img_class])
                class_counts[img_class] += 1

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

        padding = FMRI_DIMENSIONS - fmri_data.shape[1]
        fmri_data = np.pad(fmri_data, ((0, 0), (0, padding)), 'constant')

        for j0, sample_index in np.ndenumerate(sample_index_list):
            if (j0[0] + 1) % 1000 == 0:
                print(j0[0]+1)
                # break

            sample_label = fmri_labels[sample_index - 1]  # Sample label (file name)
            sample_label_num = label_table[sample_label]  # Sample label (serial number)
            class_idx, class_count = class_table[sample_label]

            # fMRI data in the sample
            sample_data = fmri_data[sample_index - 1, :]
            sample_data = np.float32(sample_data)  
            # sample_data = np.reshape(sample_data, (sample_data.size, 1, 1))  # Num voxel x 1 x 1

            # Load images
            image_file = images_table[sample_label]

            img = PIL.Image.open(image_file).convert("RGB")
            # img = img.resize(img_size, PIL.Image.BILINEAR)

            data_arr.append([])
            data_arr[-1].append(sample_data)
            data_arr[-1].append(img)
            data_arr[-1].append(class_idx) # which class is it
            data_arr[-1].append(class_count) # what index in that class is it
            data_arr[-1].append(sample_label_num)

        print("Done")
        return data_arr

class fMRIImgClassifierDataset():
    def __init__(self, dataset, split='train'):
        self.dataset = dataset
        self.set_split(split)
        
    def get_filtered(self, split):
        def f(datum):
            if split == 'train':
                return datum[3] < 7
            else:
                return datum[3] >= 7
        return f

    def set_split(self, split):
        self.split = split
        self.filtered_data = self.filter_data(self.dataset.subj_data, split)

    def filter_data(self, data, split):
        return list(filter(self.get_filtered(split), data))

    def __len__(self):
        return len(self.filtered_data)

    def __getitem__(self, idx):
        fmri, img, class_label, class_count, label = self.filtered_data[idx]
        return fmri, class_label, class_count
    

class convImgClassifierDataset():
    def __init__(self, dataset, split='train'):
        self.dataset = dataset
        self.set_split(split)
        
    def get_filtered(self, split):
        def f(datum):
            if split == 'train':
                return datum[3] < 7
            else:
                return datum[3] >= 7
        return f

    def set_split(self, split):
        self.split = split
        self.filtered_data = self.filter_data(self.dataset.subj_data, split)

    def filter_data(self, data, split):
        return list(filter(self.get_filtered(split), data))

    def __len__(self):
        return len(self.filtered_data)

    def __getitem__(self, idx):
        fmri, img, class_label, class_count, label = self.filtered_data[idx]
        return self.dataset.img_transform(img), class_label, class_count
    