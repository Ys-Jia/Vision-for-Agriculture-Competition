from data_utils import *
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import sampler
import matplotlib.image as mimg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

colors = ['black', 'white', 'springgreen', 'darkgreen', 'skyblue', 'navy', 'blue', 'red'] # contour奇怪特性,第5位置自动跳过
patchs = [mpatches.Patch(color=colors[0], label='ground'),
                  mpatches.Patch(color=colors[1], label='cloud'),
                  mpatches.Patch(color=colors[2], label='doubleplant'),
                  mpatches.Patch(color=colors[3], label='planter_skip'),
                  mpatches.Patch(color=colors[4], label='strand_water'),
                  mpatches.Patch(color=colors[6], label='water_way'),
                  mpatches.Patch(color=colors[7], label='weed_cluster')]

def plot_picture_contour(picture, masks, preds, labels=None): # picture 1*512*512*4, masks 1*512*512*1, labels 1*512*512*7
    picture = (picture * masks).cpu().numpy()
    picture_rgb = picture[:, :, :, :3].astype(dtype=np.uint8)
    picture_nir = picture[:, :, :, 3].astype(dtype=np.uint8)
    _, preds = torch.max(preds, dim=-1)
    preds = preds.cpu().numpy()
    number_pics = preds.shape[0]
    xx, yy = np.meshgrid(np.arange(0, preds.shape[1]), np.arange(0, labels.shape[2]))
    if labels is not None:
        _, labels = torch.max(labels, dim=-1)
        labels = labels.cpu().numpy()
        for i in range(number_pics):
            plt.figure(i)
            plt.subplot(1, 2, 1)
            plt.imshow(picture_rgb[i])
            plt.contourf(xx, yy, preds[i], colors=colors, alpha=0.8)
            plt.legend(handles=patchs, fontsize=8, loc='best')
            plt.subplot(1, 2, 2)
            plt.imshow(picture_rgb[i])
            plt.contourf(xx, yy, labels[i], colors=colors, alpha=0.8)
            plt.legend(handles=patchs, fontsize=8, loc='best')
    else:
        for i in range(number_pics):
            plt.figure(i)
            plt.imshow(picture_rgb[i])
            plt.contourf(xx, yy, preds[i], colors=colors, alpha=0.8)
            plt.legend(handles=patchs, fontsize=8, loc='best')
    plt.show()
    

