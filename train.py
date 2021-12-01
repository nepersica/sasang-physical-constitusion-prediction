from tqdm import tqdm
from model import *
from dataloaderV2 import BodyPartDataset

from torch.utils.data import DataLoader
import config as cfg
from utils import *
import torch.nn as nn
import wandb

import numpy as np
from score import *

from loss import *

fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)  # Tensor를 numpy로 변환
fn_denorm = lambda x, mean, std: (x * std) + mean  # DeNomarlization
fn_class = lambda x: 1.0 * (x > 0.4)

def get_data(args, mode):
    """
        Load dataset
    """
    print(f'> Getting {mode} data.... > Augmentation mode: {args.transform}')
    dataset = BodyPartDataset(root_dir=args.data_dir, transform=args.transform, image_size=args.image_size, mode=mode)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers) 

    return dataset, loader

def train(args, model, optim, device):
    """
        Train model
    """
    # write history(loss, score, etc.) on wandb 
    wandb.init(project='human parsing', reinit=True)
    wandb.run.name = 'v1'
    wandb.config = {'learning_rate':args.lr, 'epochs':args.num_epoch, 'batch_size':args.batch_size}
    wandb.watch(model)

    st_epoch = 0
    
    if args.train_continue == "on":     # load weight to continue training
        model, optim, st_epoch = load(ckpt_dir=args.ckpt_dir, model=model, optim=optim)

    
    train_dataset, train_loader = get_data(args, 'train')
    val_dataset, val_loader = get_data(args, 'val')

    num_data_train = len(train_dataset) 
    num_data_val = len(val_dataset)  

    num_batch_train = np.ceil(num_data_train / args.batch_size) 
    num_batch_val = np.ceil(num_data_val / args.batch_size)

    # Use BCE los sfunction
    criterion = nn.BCEWithLogitsLoss().to(device)

    """
        Training
    """
    for epoch in range(st_epoch + 1, args.num_epoch + 1):
        model.train() 
        total_loss_arr = []
        total_accuracy = []
        total_f1 = []
        

        for iter, data in enumerate(tqdm(train_loader), 1):
            input = data['input'].to(device)        # [N, 3, image_size, image_size]
            label = data['mask'].to(device)         # [N, image_size, image_size]
            
            output = model(input)  # [N, 256, 256]

            loss = criterion(output, label)

            optim.zero_grad()  
            loss.backward()
                        
            optim.step()

            _loss = loss.item()  
            total_loss_arr.append(_loss)

            logit = torch.sigmoid(output)
            output = logit.clone()
            output[output>0.5] = 1
            output[output<=0.5] = 0
            
            label = fn_tonumpy(label)
            output = fn_tonumpy(output)
            
            recall, precision, accuracy, f1_score, jaccard = get_score(output, label)
            
            total_accuracy.append(accuracy)
            total_f1.append(f1_score)
            
        print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                (epoch, args.num_epoch, iter, num_batch_train, np.mean(total_loss_arr)))

        wandb.log({
            "train-f1 score": np.mean(total_f1),
            "train-accuracy": np.mean(total_accuracy),
            "train-loss": np.mean(total_loss_arr)})
                
        """
        Validation
        """
        if epoch % 10 == 0:
            with torch.no_grad():
                model.eval()
                val_loss_arr = []
                total_val_accuracy = []
                total_val_f1 = []

                for iter, data in enumerate(tqdm(val_loader), 1):
                    input = data['input'].to(device)
                    label = data['mask'].to(device)

                    output = model(input)

                    loss = criterion(output, label)

                    _loss = loss.item()  
                    val_loss_arr.append(_loss)

                    logit = torch.sigmoid(output)
                    output = logit.clone()
                    output[output>0.5] = 1
                    output[output<=0.5] = 0
                    
                    label = fn_tonumpy(label)
                    output = fn_tonumpy(output)
                    
                    recall, precision, accuracy, f1_score, jaccard = get_score(output, label)

                    total_val_accuracy.append(accuracy)
                    total_val_f1.append(f1_score)

                    print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                          (epoch, args.num_epoch, iter, num_batch_train, np.mean(val_loss_arr)))

            print(f'Average Accuracy : {np.array(total_val_accuracy).mean()}')

            wandb.log({
                "val-f1 score": np.mean(total_val_f1),
                "val-accuracy": np.mean(total_val_accuracy),
                "val-loss": np.mean(val_loss_arr)})
            
        # save weight every 10 epochs
        if epoch % 10 == 0 :
            save(ckpt_dir=args.ckpt_dir, net=model, optim=optim, epoch=epoch)
            torch.save(model.state_dict(), "model.pt")
            print(f'saved model')

    print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" % (iter, num_batch_val, np.mean(loss_arr)))


