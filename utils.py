import torch
import numpy as np
import os
import config as cfg
import cv2 as cv


def denoramlize_image(input):
    mean=(0.485, 0.456, 0.406)
    mean = np.array(mean, dtype=np.float32)
    std=(0.229, 0.224, 0.225)
    std = np.array(std, dtype=np.float32)
    
    mean *= 255.0
    std *= 255.0
    
    input = (input * std) + mean
    
    return input.astype('uint8')


def to_bgr(y_pred, image_size):
    ret = np.zeros((image_size, image_size, 3), np.float32)
    for r in range(image_size):
        for c in range(image_size):
            color_id = int(y_pred[r, c])
            
            color_hex = cfg.color[color_id].lstrip('#')
            try:
                color_rgb = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4)) 
            except:
                print(color_id)
            
            ret[r, c, :] = color_rgb
    ret = ret.astype(np.uint8)
    return ret


def output_to_numpy(y_pred):
    final_output = np.zeros((y_pred.shape[0], y_pred.shape[2], y_pred.shape[3]))
    for i in range(y_pred.shape[0]):
        _output = np.argmax(_output, axis=0)
        final_output[i, ...] = _output
    return final_output


def save_predict_mask(names, inputs, y_pred, sex):
    # np_pred = output_to_numpy(y_pred)
    
    for idx in range(y_pred.shape[0]):
        name = names[idx][3:]
        input = inputs[idx]
        input = denoramlize_image(input)
        
        out = to_bgr(y_pred[idx], image_size=y_pred.shape[1])
        
        overlay_image = cv.addWeighted(input,0.4,out,0.6,0)
        
        cv.imwrite(os.path.join(f'./result/overlay/{sex}', name+'.jpg'), overlay_image)
        

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(epoch, epochs_since_improvement, model, optimizer, loss, is_best):
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'loss': loss,
             'model': model,
             'optimizer': optimizer}
    # filename = 'checkpoint_' + str(epoch) + '_' + str(loss) + '.tar'
    filename = 'checkpoint.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_checkpoint.tar')

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# 모델 저장
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
                "./%s/model_epoch%d.pth" % (ckpt_dir, epoch))
    print(f'>> save model_{epoch}.pth')



# 모델 불러오기
def load(ckpt_dir, model, optim):
    """
    Load the model trained on single GPU
    :param ckpt_dir: save path
    :param net: model
    :param optim: optimizer
    :return:
    """
    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    print(os.path.join(ckpt_dir, ckpt_lst[-1]))

    dict_model = torch.load(os.path.join(ckpt_dir, ckpt_lst[-1]))
    model.load_state_dict(dict_model['net'], strict=False)
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])
    print("Get saved weights successfully.")

    return model, optim, epoch


def load_multi(ckpt_dir, model, optim):
    """
    Load the model trained on multi GPU
    :param ckpt_dir: save path
    :param net: model
    :param optim: optimizer
    :return:
    """
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return model

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))
    pretrained_dict = dict_model['net']

    model_dict = model.state_dict()

    pretrained_dict = {k1: v for (k, v), k1 in zip(pretrained_dict.items(), model_dict)}

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])
    print("Get saved weights successfully.")

    return model, optim, epoch
