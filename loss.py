import torch.nn as nn
class DiceLoss(nn.Module):
    """
    Dice score loss function
    """
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, output, label):
        assert output.size() == label.size()
        output = output[:, 0].contiguous().view(-1)
        label = label[:, 0].contiguous().view(-1)
        intersection = (output * label).sum()
        dsc = (2. * intersection + self.smooth) / (output.sum() + label.sum() + self.smooth)

        return 1. - dsc