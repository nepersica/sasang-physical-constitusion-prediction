from tqdm import tqdm
from model import *
from dataloader import BodyPartDataset

from torch.utils.data import DataLoader
import config as cfg
from utils import *
import torch.nn as nn

import numpy as np

fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)  # Tensor를 numpy로 변환
fn_denorm = lambda x, mean, std: (x * std) + mean  # DeNomarlization
fn_class = lambda x: 1.0 * (x > 0.4)

def get_data(args, mode):
    ### Test Dataset 가져오기
    print(f'> Getting {mode} data.... > Augmentation mode: {args.transform}')
    dataset = BodyPartDataset(root_dir=args.data_dir, transform=args.transform, image_size=args.image_size, mode=mode)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers) 

    return dataset, loader

def test(args, model, optimizer, device):
    ### Test Dataset 가져오기
    test_dataset, test_loader = get_data(args, 'test')
    
    num_data_test = len(test_dataset) 

    criterion = nn.CrossEntropyLoss().to(device)
    
    st_epoch = 0
    
    
    with torch.no_grad():
        model.eval()

        loss_arr = []
        total_acc = []


        for iter, data in enumerate(tqdm(test_loader), 1):
            # forward pass
            name = data['name']
            input = data['input'].to(device)        
            label = data['mask'].to(device)         
            
            output = model(input)['out']

            loss = criterion(output, label)
            
            acc = accuracy(output, label, image_size=args.image_size)
            total_acc.append(acc)
            
            input = fn_tonumpy(input)
            save_predict_mask(name, input, output)
            

    print(f'Average Accuracy : {np.array(total_acc).mean()}')
    
if __name__ == '__main__':
    test()