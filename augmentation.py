import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

class Augmenter():
    def __init__(self, transform="on"):
        self.do_transform = transform
        
        self.normal_augment = A.Compose([
            A.Normalize(),
        ])

        self.random_transform = A.Compose([
            A.HorizontalFlip(p=1),
            A.VerticalFlip(p=1),
            A.RandomRotate90(p=1),
            A.Normalize(),
        ])

    def _to_tensor(self, name, input, mask):

        input = input.astype('float32')
        mask = mask.astype('float32')

        input = np.transpose(input, (2, 0, 1))
        # mask = np.transpose(mask, (2, 0, 1))
        
        input = torch.from_numpy(input)
        mask = torch.from_numpy(mask).type(torch.LongTensor)
        # mask = mask.reshape((1, mask.shape[0], mask.shape[1]))

        data = {'name': name, 'input': input, 'mask': mask}

        return data


    def _default_augment(self, data):
        name = data['name']
        input = data['input']
        mask = data['mask']
        
        if self.do_transform == "on":
            augmented = self.random_transform(image=input, mask=mask)
        else:
            augmented = self.normal_augment(image=input, mask=mask)

        transformed_input = augmented['image']
        transformed_mask = augmented['mask']

        data = self._to_tensor(name, transformed_input, transformed_mask)
        
        return data

