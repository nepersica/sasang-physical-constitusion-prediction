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
    for r in range(320):
        for c in range(320):
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
        _output = y_pred.cpu().numpy()[i] 
        _output = np.argmax(_output, axis=0)
        final_output[i, ...] = _output
    return final_output


def save_predict_mask(names, inputs, y_pred):
    np_pred = output_to_numpy(y_pred)
    
    for idx in range(np_pred.shape[0]):
        name = names[idx]
        input = inputs[idx]
        input = denoramlize_image(input)
        
        out = to_bgr(np_pred[idx], image_size=np_pred.shape[1])
        
        overlay_image = cv.addWeighted(input,0.4,out,0.6,0)
        
        cv.imwrite(os.path.join('./result/predict', name+'.jpg'), overlay_image)
        

def accuracy(predicts, targets, image_size, k=1):
    batch_size = targets.size(0)
    _, ind = predicts.topk(k, 1, True, True)
    ind = torch.squeeze(ind, dim=1)
    correct = ind.eq(targets)
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size / image_size / image_size)


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