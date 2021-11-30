from segmentation_models_pytorch.deeplabv3.model import DeepLabV3Plus
from segmentation_models_pytorch import Unet
import config as cfg

import config as cfg
import torch
import torch.nn as nn
from torchsummary import summary
from torchvision import models

def build_model(model_name, lr):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_name == 'deeplab':
        
        # model = models.segmentation.deeplabv3_resnet50(pretrained=False, progress=True, num_classes=cfg.seg_num).to(device)
        
        # summary(model, input_size=(3, 320, 320))
        # model = DeepLabV3Plus('resnet50', encoder_weights=None, in_channels=3, classes=cfg.seg_num).to(device)
        model = Unet(encoder_name="efficientnet-b3", encoder_weights=None, in_channels=3, classes=cfg.seg_num).to(device)
        
    # Multi GPU
    if (device.type == 'cuda') and (torch.cuda.device_count() > 1):
        print(f'Multi GPU MODE on... {torch.cuda.device_count()}')
        model = nn.DataParallel(model)
    
    # Optimizer Adam 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    return model, optimizer
        
    
    