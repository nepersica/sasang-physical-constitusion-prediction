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
        net.eval()

        loss_arr = []
        total_recall = []
        total_precision = []
        total_f1_score = []
        total_jaccard = []

        patient_score = {}

        for iter, data in enumerate(tqdm(test_loader), 1):
            # forward pass
            input = data['input'].to(device)        ]
            label = data['mask'].to(device)         
            
            output = model(input)
            
            output = output.cpu().numpy()[0]
            
            input = fn_tonumpy(input)
            label = fn_tonumpy(label)
            output = fn_tonumpy(output)
            
            output = np.argmax(output, axis=0)
            
            

            # loss = focal_loss(output, label)
            loss = criterion(output, label)
            acc = accuracy(output, label)
            

    print(f'Average Recall : {np.array(total_recall).mean()}')
    print(f'Average Precision : {np.array(total_precision).mean()}')
    print(f'Average F1 Score : {np.array(total_f1_score).mean()}')
    print(f'Average Jaccard : {np.array(total_jaccard).mean()}')

    print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" %
          (iter, num_data_test, np.mean(loss_arr)))

if __name__ == '__main__':


        with torch.no_grad():
            out = model(x_test)['out']

        out = out.cpu().numpy()[0]
        out = np.argmax(out, axis=0)
        out = to_bgr(out)

        ret = image * 0.6 + out * 0.4
        ret = ret.astype(np.uint8)

        if not os.path.exists('images'):
            os.makedirs('images')

        cv.imwrite('images/{}_image.png'.format(i), image)
        cv.imwrite('images/{}_merged.png'.format(i), ret)
        cv.imwrite('images/{}_out.png'.format(i), out)