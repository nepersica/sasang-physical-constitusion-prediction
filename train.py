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
    ### Train Dataset 가져오기
    print(f'> Getting {mode} data.... > Augmentation mode: {args.transform}')
    dataset = BodyPartDataset(root_dir=args.data_dir, transform=args.transform, image_size=args.image_size, mode=mode)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers) 

    return dataset, loader


def train(args, model, optim, device):
    
    ### Train Dataset 가져오기
    train_dataset, train_loader = get_data(args, 'train')
    ### Valid Dataset 가져오기
    val_dataset, val_loader = get_data(args, 'val')

    num_data_train = len(train_dataset) 
    num_data_val = len(val_dataset)  

    num_batch_train = np.ceil(num_data_train / args.batch_size)  # 한 배치에 학습시키는 데이터 크기
    num_batch_val = np.ceil(num_data_val / args.batch_size)

    criterion = nn.BCEWithLogitsLoss().to(device)
    
    st_epoch = 0

    for epoch in range(st_epoch + 1, args.num_epoch + 1):
        model.train()  # 모델 학습 모드
        total_loss_arr = []
        total_accuracy = []
        

        for iter, data in enumerate(tqdm(train_loader), 1):
            # forward pass
            input = data['input'].to(device)        # [N, 3, image_size, image_size]
            label = data['mask'].to(device)         # [N, image_size, image_size]
            
            # Forward prop.
            output = model(input)['out']  # [N, 320, 320]

            # Calculate loss
            loss = criterion(output, label)

            # for metrics
            logit = torch.sigmoid(output)
            output = logit.clone()
            output[output>0.5] = 1
            output[output<=0.5] = 0
            
            label = fn_tonumpy(label)
            output = fn_tonumpy(output)
            
            # acc = accuracy(output, label, args.image_size)
            # total_accuracy.append(acc)
            recall, precision, accuracy, f1_score, jaccard = get_score(output, label)

            optim.zero_grad()  
            loss.backward()
            
            clip_gradient(optim, cfg.grad_clip)
            
            optim.step()

            _loss = loss.item()  # loss 손실값 불러오기
            total_loss_arr.append(_loss)

            
        print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                (epoch, args.num_epoch, iter, num_batch_train, np.mean(total_loss_arr)))

        effective_lr = get_learning_rate(optim)
        print('Current effective learning rate: {}\n'.format(effective_lr))
        
        """
        Validation
        """
        if epoch % 100 == 0:
            with torch.no_grad():
                model.eval()
                val_loss_arr = []
                total_val_accuracy = []

                for iter, data in enumerate(tqdm(val_loader), 1):
                    # forward pass
                    input = data['input'].to(device)
                    label = data['mask'].to(device)

                    output = model(input)

                    # loss = focal_loss(output, label)
                    loss = criterion(output, label)
                    acc = accuracy(output, label)

                    _loss = loss.item()  # loss 손실값 불러오기
                    val_loss_arr.append(_loss)
                    
                    total_val_accuracy.append(acc)

                    print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                          (epoch, args.num_epoch, iter, num_batch_train, np.mean(loss_arr)))

            print(f'Average Accuracy : {np.array(total_val_accuracy).mean()}')

            wandb.log({
                "val-accuracy": np.mean(total_val_accuracy),
                "val-loss": np.mean(val_loss_arr)})
            
        if epoch % 10 == 0 :
            save(ckpt_dir=args.ckpt_dir, net=model, optim=optim, epoch=epoch)
            torch.save(model.state_dict(), "model.pt")
            print(f'saved model')

    print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" % (iter, num_batch_val, np.mean(loss_arr)))


