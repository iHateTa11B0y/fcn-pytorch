import torch
import cv2
from collections import OrderedDict
import numpy as np

def crop_weight(wf):
    a = torch.load(wf)
    b = OrderedDict([])
    for k in list(a.keys()):
        if k[:7] == 'module.':
            b[k[7:]] = a[k]
        else:
            b[k] = a[k]
    return b

def to_binary_mask(output, dsize, thresh=0.5):
    binary_mask = output.cpu() > thresh
    binary_mask = binary_mask.float().numpy()
    binary_mask = cv2.resize(binary_mask, dsize=dsize, interpolation=cv2.INTER_LINEAR)
    return binary_mask.astype(np.uint8)


