import numpy as np
import torch
from torch import nn
from torch import functional as F
from torch.cuda import amp
import matplotlib.pyplot as plt
from data_utils import *
from layer_utils import *
from data_utils import *
from plot_utils import *

means = torch.tensor([[[[116.1301, 118.6523, 121.8633, 108.1013]]]], device=device, dtype=dtype)
stds = torch.tensor([[[[45.3765, 42.2304, 41.7526, 44.9108]]]], device=device, dtype=dtype)


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, scores, labels, eps=1e-8):
        num_pics = scores.shape[0]
        logits = torch.sigmoid(scores)
        preds = nn.functional.threshold(1 - nn.functional.threshold(1-logits, 0.5, 0), 0.5, 0)
        intersection = torch.sum(preds * labels)
        union = torch.sum(preds) + torch.sum(labels)
        dice = intersection * 2 / (union + eps) / num_pics
        return 1 - dice

class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, pos_weight=None):
        super().__init__()
        if pos_weight is None: pos_weight = torch.ones(7).to(device=device, dtype=dtype)
        if weight is None: weight = torch.ones(7).to(device=device, dtype=dtype)
        self.BceLoss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, weight=weight)
        self.DiceLoss = DiceLoss()
    
    def forward(self, scores, labels):
        Bceloss_item = self.BceLoss(scores, labels)
        Diceloss_item = self.DiceLoss(scores, labels)
        return 0.5 * Bceloss_item + 0.5 * Diceloss_item


class Solver(object):
    def __init__(self, model, optimizer, dataloader, val_loader, batch_time=1, loss_name='BCE', epochs=1, print_every=1, verbose=True):
        self.model = model
        self.optimizer = optimizer
        self.loss_name = loss_name
        self.dataloader = dataloader
        self.val_loader = val_loader
        self.batch_time = batch_time
        self.epochs = epochs

        self.print_every = print_every
        self.verbose = verbose

        self._initialize()

    def _initialize(self):
        self.scaler = amp.GradScaler(enabled=(device is 'cuda'))
        self.loss_history = []
        self.val_acc = []
        self.best_parameters = {}
        self.classes = ['ground', 'cloud', 'doubleplant', 'planter_skip', 'strand_water', 'water_way', 'weed_cluster']
        self.best_acc = 0
        self.check_point = 500
        self.postive_weight = torch.tensor([10, 10, 10, 10, 10, 10, 10], device=device)
        self.weight = torch.tensor([1e-3, 1, 1, 1, 1, 1, 1], device=device)

    def _save(self, save_path):
        package = {}
        package['model'] = self.model
        package['optimizer'] = self.optimizer
        package['loss_history'] = self.loss_history
        package['val_acc'] = self.val_acc
        package['best_parameters'] = self.best_parameters
        package['best_acc'] = self.best_acc
        package['postive_weight'] = self.postive_weight
        package['weight'] = self.weight
        torch.save(package, save_path)
        print('Sucessful Save')

    def _load(self, package):
        self.model = package['model']
        self.optimizer = package['optimizer']
        self.loss_history = package['loss_history']
        self.val_acc = package['val_acc']
        self.best_parameters = package['best_parameters']
        self.best_acc = package['best_acc']
        self.postive_weight = package['postive_weight']
        self.weight = package['weight']
        print('Save load!')

    def train(self):
        iterations = len(self.dataloader)
        full_iterations = self.epochs * iterations
        loss_box = []
        print('*****start to train!*****')
        print('Epochs-->', self.epochs, 'iterations-->', full_iterations)
        if self.loss_name == 'BCE':
            self.loss = nn.BCEWithLogitsLoss(weight=self.weight, pos_weight=self.postive_weight)
        if self.loss_name == 'BCEDICE':
            self.loss = BCEDiceLoss(weight=self.weight, pos_weight=self.postive_weight)
        for epoch in range(self.epochs):
            print('Epoch------(%d / %d)-------' %(epoch, self.epochs))
            self.optimizer.zero_grad()
            for t, all_data in enumerate(self.dataloader):
                self.model.train()
                train_data, masks, labels = concencate_data(all_data, means, stds, preprocess=True)
                X = (train_data * masks).transpose(1, 3)
                del train_data, masks # DEL目的减小中间变量来少缓存
                with amp.autocast(enabled=(device is 'cuda')): # 开启混合精度模式，增快收敛和减少显存
                  scores = self.model(X)
                  loss = self.loss(scores, labels)

                self.scaler.scale(loss).backward()
                loss_box.append(loss.item())
                if (t+1) % self.batch_time == 0:
                    self.scaler.step(self.optimizer) # 更新梯度
                    self.scaler.update() # update防止梯度出现nan或空则不更新
                    self.optimizer.zero_grad()
                    true_loss, loss_box = np.mean(loss_box), []
                    self.loss_history.append(true_loss)
                    print('Iteration(%d / %d)-->loss:%.5f' %(t+1, iterations, true_loss))

                if (t+1) % self.print_every == 0:
                    mIOU, mean_mIOU = self.check_accuracy()
                    self.val_acc.append(mIOU)
                    if mean_mIOU > self.best_acc:
                        self.save_best()
                        self.best_acc = mean_mIOU
                        print('Now best mean_mIOU is:', mean_mIOU)

                if (t+1) % self.check_point == 0:
                    print('Checkpoint:%d, Save!!'%(t+1))
                    save_path = '/content/drive/MyDrive/Vison-for-Agriculture-project/Term_project_py/model_8_10.pth'
                    self._save(save_path)

    def check_accuracy(self, loader=None, plot_every=None):
        if loader is None: loader = self.val_loader
        correct_labels, denominator, label_all, pixels = 0, 0, 0, 0
        iteration_all = len(loader)
        picture_number = len(loader.sampler)
        if plot_every is None: plot_every = iteration_all
        with torch.no_grad():
            for t, all_data in enumerate(loader):
                self.model.eval()
                val_data, masks, labels = concencate_data(all_data, means, stds, preprocess=True)
                plot_val_data = val_data * stds + means
                real_val = (val_data * masks).transpose(1, 3)
                out = self.model(real_val)
                del real_val
                max_pred, index = torch.max(out, dim=-1) # 0 is value，1 is index，no need argmax
                max_pred = max_pred.view(*(max_pred.shape), 1)
                preds = (out - max_pred) >= 0
                correct_labels += torch.sum(preds * labels, dim=(0, 1, 2))
                denominator += torch.sum((preds + labels)>0, dim=(0, 1, 2))
                label_all += torch.sum(labels, dim=(0, 1, 2))
                pixels += np.prod(labels.shape[0:3])
                if (t+1) % plot_every == 0:
                    plot_picture_contour(plot_val_data, masks, preds, labels)

            mIOU = (correct_labels / (denominator+1e-8) * 100).cpu().numpy()
            each_label_percentage = (label_all / pixels * 100).cpu().numpy()
            print('MIOU for each:', end=' ')
            for i in range(len(mIOU)):
                print(self.classes[i]+':'+str(mIOU[i])+'%', end=' ')
            print()
            print('Labl for each:', end=' ')
            for i in range(len(mIOU)):
                print(self.classes[i]+':'+str(each_label_percentage[i])+'%', end=' ')
            print()
            label_than_0 = np.sum(label_all.cpu().numpy() > 0)
            mean_mIOU = np.sum(mIOU) / label_than_0
            print('Mean MIOU:', mean_mIOU, '%')

            return mIOU, mean_mIOU

    def prediction(self, pic_data, masks):
      with torch.no_grad():
        self.model.eval()
        test_data, masks = concencate_data(pic_data, means, stds, preprocess=True, test=True)
        
        real_test = (test_data * masks).transpose(1, 3)
        out = self.model(real_test)
        preds = torch.argmax(out, dim=-1)
        return preds # 512 * 512 * 1

    def save_best(self):
        for k, v in self.model.state_dict().items():
            self.best_parameters[k] = torch.clone(v)

    def set_best(self, model=None):
        if model is None: model = self.model; print('Directly set on self.model')
        model.load_state_dict(self.best_parameters)
        return model
