from tqdm import tqdm
from model import *
from dataloaderV2 import BodyPartDataset

from torch.utils.data import DataLoader
import config as cfg
from utils import *
import torch.nn as nn

import numpy as np
from score import *

fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)  # Tensor를 numpy로 변환
fn_denorm = lambda x, mean, std: (x * std) + mean  # DeNomarlization
fn_class = lambda x: 1.0 * (x > 0.4)

def get_data(args, mode):
    print(f'> Getting {mode} data.... > Augmentation mode: {args.transform}')
    dataset = BodyPartDataset(root_dir=args.data_dir, transform=args.transform, image_size=args.image_size, mode=mode)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers) 

    return dataset, loader

def test(args, model, optimizer, device):
    test_dataset, test_loader = get_data(args, 'test')
    num_data_test = len(test_dataset) 

    model, optimizer, st_epoch = load(ckpt_dir=args.ckpt_dir, model=model, optim=optimizer)
    print(f'Get lr:{args.lr}, epoch:{st_epoch}.pth. \n')

    st_epoch = 0
    
    with torch.no_grad():
        model.eval()

        total_accuracy = []
        total_f1 = []
        total_precision = []
        total_recall = []
        total_jaccard = []


        for iter, data in enumerate(tqdm(test_loader), 1):
            # forward pass
            name = data['name']
            input = data['input'].to(device)        
            label = data['mask'].to(device)         
            
            output = model(input)

            logit = torch.sigmoid(output)   
            output = logit.clone()
            output[output>0.5] = 1
            output[output<=0.5] = 0
            
            input = fn_tonumpy(input)
            label = fn_tonumpy(label)
            output = fn_tonumpy(output)
            
            recall, precision, accuracy, f1_score, jaccard = get_score(output, label)
            
            save_predict_mask(name, input, output, sex='female')
            
            total_accuracy.append(accuracy)
            total_f1.append(f1_score)
            total_precision.append(precision)
            total_recall.append(recall)
            total_jaccard.append(jaccard)
            

    print(f'Average Accuracy : {np.array(total_accuracy).mean()}')
    print(f'Average f1 : {np.array(total_f1).mean()}')
    print(f'Average Precision : {np.array(total_precision).mean()}')
    print(f'Average Recall : {np.array(total_recall).mean()}')
    print(f'Average Jaccard : {np.array(total_jaccard).mean()}')
    
if __name__ == '__main__':
    test()