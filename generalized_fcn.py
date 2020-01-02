import torch
from torch import nn
from unet import Unet
from torch.nn import functional as F

class GeneralizedFCN(nn.Module):
    def __init__(self):
        super(GeneralizedFCN, self).__init__()
        self.body = Unet(3,64,1)
        
    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        
        res = self.body(images)

        if self.training:
            # TODO: add mask matching strategy for multiple masks
            #targets_m = []
            #for m in targets.get_field('masks'):
            #    targets_m.append(m.get_mask_tensor())
            loss = F.binary_cross_entropy_with_logits(res, targets)
            return {'mask_loss': loss}

        return res
        
