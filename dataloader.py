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

# Tensor shape: (15, hegith, width)

class BodyPartDataset(Dataset):
    def __init__(self, root_dir, transform="on", image_size=None, mode='train'):
        self.root_dir = root_dir
        self.image_size = image_size
        self.transform = Augmenter(transform)
        self.mode=mode        
        self.dataset = self._get_data_names()
    
    def _get_data_names(self):
        self.data_list = []
        image_dirs = open(f"./dataset/v2/{args.mode}.txt",'r').read().splitlines()
        
        for dir in image_dirs:
            image_list = os.listdir(os.path.join(self.root_dir, 'image', dir))
            for image_name in image_list:
                name = image_name[:-4]
                image_path = os.path.join(self.root_dir, 'image', dir, image_name)
                label_path = os.path.join(self.root_dir, 'label', dir, name + '.json')
                
                self.data_list.append({'name': name, 'image_path':image_path, 'label_path':label_path})
                
        num_train = int(len(self.data_list)*0.8)
        num_rest = len(self.data_list)-num_train

        if num_rest % 2 == 1:
            num_val = int(num_rest/2)+1
        else:
            num_val = int(num_rest/2)
            
        if self.mode == 'train':
            dataset = self.data_list[:num_train]
        elif self.mode == 'val':
            dataset = self.data_list[num_train:num_train+num_val]
        elif self.mode == 'test':
            dataset = self.data_list[num_train+num_val:]
        return dataset

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
        mask = self._load_json(label_path, (input.shape[0], input.shape[1]), input)
        
        dim = (self.image_size, self.image_size)
        resized_input = cv.resize(input, dim, interpolation = cv.INTER_AREA)
        resized_mask = cv.resize(mask, dim, interpolation = cv.INTER_NEAREST)
        
        data = {'name': label_path[26:-5], 'input': resized_input, 'mask': resized_mask}
        
        return data
        
        
    def _load_json(self, label_path, shape, image):
        total_mask = np.zeros((shape[0], shape[1], cfg.seg_num))
        mask = np.zeros((shape[0], shape[1], 3))
        
        with open(label_path, 'r', encoding='UTF8') as f:
            content = json.load(f)
        annotations = content['labelingInfo']
        for annotation in annotations:
            label = annotation['polygon']['label']            
            x = list(map(int, annotation['polygon']['location'].split()[0::2]))
            y = list(map(int, annotation['polygon']['location'].split()[1::2]))
            location = np.array([list(e) for e in zip(x, y)])
            
            if label == '위아랫쪽다리' :
                label = '아래왼쪽다리'
            
            mask, index = self._fill_mask(label, location, (total_mask.shape[0], total_mask.shape[1]), label_path)
            total_mask[:, :, index] = mask
            
            # color_hex = annotation['polygon']['color'].lstrip('#')
            # color_rgb = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))
            # mask = fillPoly(mask, pts = [location], color=color_rgb)            
        # overlay_image = cv.addWeighted(image,0.4,mask.astype('uint8'),0.6,0)
        # cv.imwrite(os.path.join('./result/overlay', label_path[24:-5]+'.jpg'), overlay_image)
        
        total_mask = np.argmax(total_mask, axis=2)
        
        # _total_mask = np.zeros((shape[0], shape[1], 3))
        # _total_mask[:,:,0] = total_mask
        # _total_mask[:,:,1] = total_mask
        # _total_mask[:,:,2] = total_mask
        
        
    
        return total_mask
            
                   
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

    for i in tqdm(range(6835, len(dataset))):
        dataset.__getitem__(i)