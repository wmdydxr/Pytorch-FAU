"""
Load the BP4D or the DISFA dataset.
"""
import os
import cv2
import numpy as np
import torch
import inspect
from torch.utils.data import Dataset

class MyDatasets(Dataset):
    def __init__(self, sigma=2, size=256, heatmap=32,AU_positions=10, database=''):
        if database == 'BP4D':
            txt_file = open('/data0/evelyn/BP4D_AUCoding/train_BP4D.txt','r')
        if database == 'DISFA':
            txt_file = open('/data0/evelyn/DISFA/train_DISFA.txt','r')
        if database == 'BP4D-val':
            txt_file = open('/data0/evelyn/BP4D_AUCoding/val_BP4D.txt','r')
        if database == 'DISFA-val':
            txt_file = open('/data0/evelyn/DISFA/test_DISFA.txt','r')
        if database == 'demo':
            txt_file = open('./data/examples.txt','r')
        lines = txt_file.readlines()[0::]
        names = [l.split()[0] for l in lines]
        coords = [l.split()[1::] for l in lines] 
        self.database = database      
        self.data = dict(zip(names,coords))
        self.imgs = list(set(names))
        self.len = len(self.imgs)

    def generate_target(self, points, intensity):
        target = np.zeros((self.AU_positions,self.heatmap,self.heatmap),dtype=np.float32)
        gs_range = self.sigma * 15
        for point_id in range(self.AU_positions):
            feat_stride = self.size / self.heatmap
            mu_x = int(points[point_id][0] / feat_stride + 0.5)
            mu_y = int(points[point_id][1] / feat_stride + 0.5)
            ul = [int(mu_x - gs_range), int(mu_y - gs_range)]
            br = [int(mu_x + gs_range + 1), int(mu_y + gs_range + 1)]
            x = np.arange(0, 2*gs_range+1, 1, np.float32)
            y = x[:, np.newaxis]
            x_center = y_center = (2*gs_range+1) // 2
            g = np.exp(- ((x - x_center) ** 2 + (y - y_center) ** 2) / (2 * self.sigma ** 2))
            g_x = max(0, -ul[0]), min(br[0], self.heatmap) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.heatmap) - ul[1]
            img_x = max(0, ul[0]), min(br[0], self.heatmap)
            img_y = max(0, ul[1]), min(br[1], self.heatmap)
            target[point_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = intensity[point_id]*g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        return target*255.0

    def fetch(self,index):
        path_to_img = self.imgs[index]
        image = cv2.cvtColor(cv2.imread(path_to_img), cv2.COLOR_BGR2RGB)
        if self.database == 'demo':
            return image, [0], [0] 
        AUs = self.data[self.imgs[index]]
        AUs = np.float32(self.data[self.imgs[index]]).reshape(-1,3)
        AU_coords = AUs[:,:2]
        AU_intensity = AUs[:,2]
        return image, AU_coords, AU_intensity

    def __getitem__(self,index):
        image, AU_coords, AU_intensity = self.fetch(index)
        nimg = len(image) 
        sample = dict.fromkeys(['Im'], None)
        out = dict.fromkeys(['image','points'])
        image_np = torch.from_numpy((image/255.0).swapaxes(2,1).swapaxes(1,0))
        out['image'] = image_np.type_as(torch.FloatTensor())
        out['AU_coords'] = np.floor(AU_coords)
        if not self.database == 'demo':                 
            target = self.generate_target(out['AU_coords'], AU_intensity)
            target = torch.from_numpy(target).type_as(torch.FloatTensor())
            sample['target'] = target
            sample['pts'] = out['AU_coords']
            sample['intensity'] = AU_intensity
        sample['Im'] = out['image']
        return sample

    def __len__(self):
        return len(self.imgs)


  
    