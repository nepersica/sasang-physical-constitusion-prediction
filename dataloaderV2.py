import os
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from cv2 import fillPoly
import cv2 as cv
from tqdm import tqdm
import config as cfg
import matplotlib.pyplot as plt
from augmentation import Augmenter
import torch

# 모델은
# 1)segmentation정보만 가지고 
# 2) 수치 정량 데이터만 가지고 
# 3)두 데이터 모두 가지고 해봐도 좋을것 같고,
# 4)기존 한의학적 이론을 바탕으로 rule base로 모델링 한것과 비교

class BodyPartDataset(Dataset):
    def __init__(self, root_dir, transform="on", image_size=None, mode='train'):
        self.root_dir = root_dir
        self.image_size = image_size
        self.transform = Augmenter(transform)
        self.mode=mode    
        self.dataset = self._get_data_names()
    
    def _get_data_names(self):
        data_list = []
        image_dirs = open(f"./dataset/v2/{self.mode}.txt",'r').read().splitlines()
        print(self.root_dir)
        # image_dirs = os.listdir(os.path.join(self.root_dir, 'image'))
        for dir in image_dirs:
            image_list = os.listdir(os.path.join(self.root_dir, 'image', dir))
            for image_name in image_list:
                name = image_name[:-4]
                if self.mode == 'test':
                    if not name[-2:] == '03':
                        continue
                image_path = os.path.join(self.root_dir, 'image', dir, image_name)
                label_path = os.path.join(self.root_dir, 'label', dir, name + '.json')
                                
                data_list.append({'name': name, 'image_path':image_path, 'label_path':label_path})
          
        return data_list

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self._load_data(index)
        data = self.transform._default_augment(data)
        
        return data
          
        
    def _load_data(self, index):            
        image_path = self.dataset[index]['image_path']
        input = cv.imread(image_path)

        label_path = self.dataset[index]['label_path']
        try:
            mask = self._load_json(label_path, (input.shape[0], input.shape[1]), input)
        except:
            mask = np.zeros((input.shape[0], input.shape[1]))
        
        dim = (self.image_size, self.image_size)
        resized_input = cv.resize(input, dim, interpolation = cv.INTER_AREA)
        resized_mask = cv.resize(mask, dim, interpolation = cv.INTER_NEAREST)
        
        data = {'name': label_path[26:-5], 'input': resized_input, 'mask': resized_mask}
        
        return data
        
        
    def _load_json(self, label_path, shape, image):
        mask = np.zeros((shape[0], shape[1], 3))
        
        with open(label_path, 'r', encoding='UTF8') as f:
            content = json.load(f)
        annotations = content['labelingInfo']
        for annotation in annotations:
            label = annotation['polygon']['label']            
            x = list(map(int, annotation['polygon']['location'].split()[0::2]))
            y = list(map(int, annotation['polygon']['location'].split()[1::2]))
            location = np.array([list(e) for e in zip(x, y)])
            
            if not label == '몸통':
                continue
            
            # mask, index = self._fill_mask(label, location, shape, label_path)
            # break
            color_hex = annotation['polygon']['color'].lstrip('#')
            color_rgb = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))
            mask = fillPoly(mask, pts = [location], color=color_rgb)            
        overlay_image = cv.addWeighted(image,0.4,mask.astype('uint8'),0.6,0)
        
        if label_path[35:-5] == '03':
            cv.imwrite(os.path.join('./result/overlay/origin', label_path[24:-5]+'.jpg'), overlay_image)
        
        return mask
            
                   
    def _fill_mask(self, label, location, shape, label_path):
            
        mask = np.zeros((shape[0], shape[1]))
        mask = fillPoly(mask, pts = [location], color=255)
        mask /= 255      
        
        try:
            index = cfg.part_KOR.index(label)
        except:
            print(label)
            print(label_path)
            visualize(mask)
            exit(-1)
        
        return mask, index

def visualize(self, image):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    from main import get_args
    args = get_args()
    
    dataset = BodyPartDataset(root_dir=args.data_dir, transform="on", image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers) 

    for i in tqdm(range(len(dataset))):
        dataset.__getitem__(i)