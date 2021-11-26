import argparse
import torch.nn as nn
import torch
from model import build_model
from train import train
from test import test

# https://github.com/foamliu/Look-Into-Person-PyTorch/blob/ca524bac51e3b54a0d723e746ee400905567adcb/train.py#L97

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_args():
    parser = argparse.ArgumentParser(description="Train the UNet",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--lr", default=1e-4, type=float, dest="lr")
    parser.add_argument("--batch_size", default=2, type=int, dest="batch_size")
    parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")

    parser.add_argument("--transform", default="off", type=str, dest="transform")
    parser.add_argument("--image_size", default=320, type=int, dest="image_size")

    parser.add_argument("--data_dir", default="./dataset/v2", type=str, dest="data_dir")
    parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")
    parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
    parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
    parser.add_argument("--pretrained_dir", default="./pretrained", type=str, dest="pretrained_dir")
    parser.add_argument("--num_workers", default=4
                        , type=int, dest="num_workers")

    parser.add_argument("--mode", default="test", type=str, dest="mode")
    parser.add_argument("--model", default="deeplab", type=str, dest="model")
    parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")
    parser.add_argument("--pretrained", default="off", type=str, dest="pretrained")
    parser.add_argument("--transfer", default="off", type=str, dest="transfer")


    args = parser.parse_args()

    return args

def main(args):
    model, optimizer = build_model(args.model, args.lr)
    
    if args.mode == 'train':  # Train 데이터, Validation 데이터
        train(args, model, optimizer, device)

    else:   # Test 데이터
        test(args, model, optimizer, device)


if __name__ == '__main__':
    args = get_args()
    main(args)

