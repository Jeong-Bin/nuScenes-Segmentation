"""
Source: https://github.com/pytorch/vision
"""
import torch
import numpy as np
import random

from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from torchvision.transforms import RandomErasing, ColorJitter

from utils.functions import seed_fixer
seed_fixer(2025)

class Resize(object):
    def __init__(self, size):
        self.size = size
    
    def __call__(self, image, mask):
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask)
        image = TF.resize(image, self.size)
        mask = TF.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)
        return image, mask
    
class TopCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size
    
    def __call__(self, image, mask):
        if isinstance(image, Image.Image):
            image = np.array(image)
        if isinstance(mask, Image.Image):
            mask = np.array(mask)

        image = image[self.crop_size:, :, :]
        mask = mask[self.crop_size:, :]

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask)

        return image, mask


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, mask):
        if random.random() < self.p:
            if isinstance(mask, np.ndarray):
                mask = Image.fromarray(mask)
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        return image, mask

class RandomColorJitter(object):
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1, p=0.5):
        self.transform = ColorJitter(brightness, contrast, saturation, hue)
        self.p = p
    
    def __call__(self, image, mask):
        if random.random() < self.p:
            image = self.transform(image)
        return image, mask

class GaussianBlur(object):
    def __init__(self, kernel_size=3, sigma=(0.1, 2.0), p=0.5):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p
    
    def __call__(self, image, mask):
        if random.random() < self.p:
            image = TF.gaussian_blur(image, self.kernel_size, self.sigma)
        return image, mask

# class RandomErasing(object):
#     def __init__(self, scale=(0.01, 0.02), ratio=(0.3, 3.3), value=1, max_nums=1, p=0.5):
#         self.transform = TF.RandomErasing(p, scale, ratio, value)
#         self.max_nums = max_nums
#         self.p = p
        
#     def __call__(self, image, mask):
#         for _ in range(self.max_nums):
#             if random.random() < self.p:
#                 image = self.transform(image)
#         return image, mask


class MultiRandomErasing(object):
    def __init__(self, scale=(0.01, 0.02), ratio=(0.3, 3.3), values=(0, 1), max_nums=1, p1=0.5, p2=0.5):
        #value = random.uniform(values[0], values[1])
        #self.transform = RandomErasing(p, scale, ratio, value)
        self.max_nums = max_nums
        self.scale = scale
        self.ratio = ratio
        self.values = values
        self.p1 = p1
        self.p2 = p2
        
    def __call__(self, image, mask):
        if random.random() < self.p1:
            for _ in range(self.max_nums):
                value = random.uniform(self.values[0], self.values[1])
                transform = RandomErasing(p=self.p2, scale=self.scale, ratio=self.ratio, value=value)
                image = transform(image)
                #image = self.transform(image)
        return image, mask
    

class ToTensor(object):
    def __call__(self, image, mask):
        image = TF.to_tensor(image)
        return image, mask

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, image, mask):
        image = TF.normalize(image, self.mean, self.std)
        return image, mask

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask