import torch
import numpy as np
import os

def accuracy(scores, targets, image_size, k=1):
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
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