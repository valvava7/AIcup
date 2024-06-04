import numpy as np
import torch 
import torch.nn as nn
import random
def seedeverything(seed = 42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
class Binarize():
    def __init__(self, threshold=0.5):
        self.threshold = threshold
    def __call__(self, img):
        return (img>self.threshold).float()
class Scale():
    def __call__(self, img):
        return img/255.1
class Clip():
    def __call__(self, img):
        img[img>1] = 1.0
        img[img<0] = 0.0
        return img

class RandomChannelSwap():
    def __init__(self, p=0.1):
        self.p = p
    def __call__(self, img):
        assert img.dim() == 3 #input should be a 3-dim tensor of shape (C,H,W)
        if random.random() < self.p:
            channels = [0, 1, 2]
            random.shuffle(channels)
            img = img[channels, :, :]
        return img
        

class add_gaussian_noise():
    def __init__(self, magnitude):
        self.mag = magnitude
    def __call__(self, img):
        img = img + torch.randn_like(img) * self.mag
        return img

class Dice_Loss(nn.Module):
    def __init__(self, beta=0.3, smooth=1):
        super().__init__()
        self.beta = beta
        self.smooth = smooth
    def forward(self, pred, target):
        #assume all element in target is in {0,1}
        pred = torch.flatten(pred, start_dim = 1)
        target = torch.flatten(target, start_dim = 1)
        intersect = torch.sum( pred * target , 1) * (1+self.beta)
        union = torch.sum(pred, 1) + self.beta * torch.sum(target, 1)
        dice_coef = (intersect+self.smooth)/(union+self.smooth)
        return 1-dice_coef.mean





        
 