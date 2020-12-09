import os, sys, torch, math
import cv2
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class OutIntensity(torch.nn.Module):
    """Infer AU intensity from a heatmap: :(x, y) = argmax H """
    def __init__(self):
        super(OutIntensity,self).__init__()
        
    def forward(self,x):
        batch_size = x.shape[0]
        num_points = x.shape[1]
        width = x.shape[2]
        x_ = x.to('cpu').detach().numpy().astype(np.float32).copy()
        heatmaps_reshaped = x_.reshape((batch_size, num_points, -1))
        intensity = heatmaps_reshaped.max(axis=2)
        return intensity/255.0
