import os
import numpy as np
import matplotlib.image as mpimg
import torch.utils.data as data
import glob
import torch
import sys

fold_path = '//content//drive//MyDrive//Vison-for-Agriculture-project//Term_project_py//vision-for-agriculture//Agriculture-Vision'
if fold_path not in sys.path:
    sys.path.append(fold_path)

dtype = torch.float64
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

class Dataset_agriculture(data.Dataset):

    def __init__(self, path):
        super().__init__() # super的作用是调用一次父类的方法，__init__()，理论上也可以用data.Dataset.__init__(self)来操作
        self.rgb_path = sorted(glob.glob(os.path.join(path, 'images', 'rgb', '*.jpg')))
        self.nir_path = sorted(glob.glob(os.path.join(path, 'images', 'nir', '*.jpg')))
        self.rgb_masks = sorted(glob.glob(os.path.join(path, 'masks', '*.png')))
        self.rgb_boundary = sorted(glob.glob(os.path.join(path, 'boundaries', '*.png')))
        self.label_cloud = sorted(glob.glob(os.path.join(path, 'labels', 'cloud_shadow', '*.png')))
        self.label_dbplant = sorted(glob.glob(os.path.join(path, 'labels', 'double_plant', '*.png')))
        self.label_planter_skip = sorted(glob.glob(os.path.join(path, 'labels', 'planter_skip', '*.png')))
        self.label_strand_water = sorted(glob.glob(os.path.join(path, 'labels', 'standing_water', '*.png')))
        self.label_water_way = sorted(glob.glob(os.path.join(path, 'labels', 'waterway', '*.png')))
        self.label_weed_cluster = sorted(glob.glob(os.path.join(path, 'labels', 'weed_cluster', '*.png')))

    def __getitem__(self, index):
        rgb_pic = mpimg.imread(self.rgb_path[index]).copy()
        nir_pic = mpimg.imread(self.nir_path[index]).copy()
        rgb_masks = mpimg.imread(self.rgb_masks[index]).copy()
        rgb_boundary = mpimg.imread(self.rgb_boundary[index]).copy()
        label_cloud = mpimg.imread(self.label_cloud[index]).copy()
        label_dbplant = mpimg.imread(self.label_dbplant[index]).copy()
        label_planter_skip = mpimg.imread(self.label_planter_skip[index]).copy()
        label_strand_water = mpimg.imread(self.label_strand_water[index]).copy()
        label_water_way = mpimg.imread(self.label_water_way[index]).copy()
        label_weed_cluster = mpimg.imread(self.label_weed_cluster[index]).copy()

        return (rgb_pic, nir_pic, rgb_masks, rgb_boundary,
            label_cloud, label_dbplant, label_planter_skip,
            label_strand_water, label_water_way, label_weed_cluster)

    def __len__(self):
        return len(self.rgb_path)

def concencate_data(all_data, means=0, stds=1, preprocess=False, to_cuda=True, test=False):
    (rgb_pic, nir_pic, rgb_masks, rgb_boundary,
    label_cloud, label_dbplant, label_planter_skip,
    label_strand_water, label_water_way, label_weed_cluster) = all_data

    train_data = torch.cat([rgb_pic, nir_pic.view(*(nir_pic.shape), 1)], dim=3)
    masks = (rgb_masks * rgb_boundary).view(*(rgb_masks.shape), 1)
    if test:
        if to_cuda: 
            test_data = train_data.to(device=device, dtype=dtype)
            masks = masks.to(device=device, dtype=torch.long) 
        return test_data, masks
    
    list_labels = [label_cloud, label_dbplant, label_planter_skip, label_strand_water, label_water_way, label_weed_cluster]
    for i in range(len(list_labels)):
        list_labels[i] = list_labels[i].view(*(label_cloud.shape), 1)
    six_labels = torch.cat(list_labels, dim=-1)
    ground_label = (torch.sum(six_labels, dim=-1) == 0).view(*(label_cloud.shape), 1)
    full_labels = torch.cat([ground_label, six_labels], dim=-1)

    if to_cuda is True:
        train_data = train_data.to(device=device, dtype=dtype)
        masks = masks.to(device=device, dtype=torch.long)
        full_labels = full_labels.to(device=device, dtype=dtype)

    if preprocess: train_data = (train_data - means) / stds

    return train_data, masks, full_labels

def compute_(data_loader):
    length_all = len(data_loader.sampler)
    numerical_stable, times = 1e-10, 1
    sum_rgb, sum_nir, pixels, process = 0, 0, 0, 0
    sum_std_rgb, sum_std_nir = 0, 0
    # num_samples = len(data_loader) # len是iteration的次数
    print('Start compute mean!')
    for all_data in data_loader:
        rgb_pic, nir_pic = all_data[0], all_data[1]
        rgb_pic = rgb_pic.to(device=device, dtype=dtype);
        nir_pic = nir_pic.to(device=device, dtype=dtype)
        sum_rgb += torch.sum(rgb_pic, dim=(0, 1, 2)) * numerical_stable
        sum_nir += torch.sum(nir_pic) * numerical_stable
        process += rgb_pic.shape[0]
        percentage = process / length_all * 100
        if percentage - 10 * times >= 0: print('Process:', percentage, '%'); times += 1

    pixels = int(length_all * rgb_pic.shape[1] * rgb_pic.shape[2])
    rgb_mean = sum_rgb / pixels / numerical_stable
    nir_mean = sum_nir / pixels / numerical_stable

    process, times = 0, 1
    print('Start compute std!')
    for all_data in data_loader:
        rgb_pic, nir_pic = all_data[0], all_data[1]
        rgb_pic = rgb_pic.to(device=device, dtype=dtype);
        nir_pic = nir_pic.to(device=device, dtype=dtype)
        sum_std_rgb += torch.sum(torch.square(rgb_pic - rgb_mean.view(1, 1, 1, 3)), dim=(0, 1, 2)) * numerical_stable
        sum_std_nir += torch.sum(torch.square(nir_pic - nir_mean.view(1, 1, 1))) * numerical_stable
        process += rgb_pic.shape[0]
        percentage = process / length_all * 100
        if percentage - 10 * times >= 0: print('Process:', percentage, '%'); times += 1

    rgb_std = torch.sqrt(sum_std_rgb / pixels / numerical_stable)
    nir_std = torch.sqrt(sum_std_nir / pixels / numerical_stable)

    means = torch.cat([rgb_mean.view(1, 1, 1, 3), nir_mean.view(1, 1, 1, 1)], dim=-1)
    stds = torch.cat([rgb_std.view(1, 1, 1, 3), nir_std.view(1, 1, 1, 1)], dim=-1)

    return means, stds