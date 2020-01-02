import torch
import numpy as np
from generalized_fcn import GeneralizedFCN
import cv2
from transforms import build_transforms
from collections import OrderedDict

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
    return binary_mask
 
class Infer(object):
    def __init__(self, model_path):
        super(Infer, self).__init__()
        model = GeneralizedFCN()
        checkpoint = crop_weight(model_path)
        model.load_state_dict(checkpoint)
        model.eval()
        
        self.model = model
        self.transforms = build_transforms()

    def eval(self, image_arr):
        img_ori = image_arr
        dshape = (img_ori.shape[1], img_ori.shape[0])
        img = img_ori.copy()
        img = self.transforms(img)
        output = self.model(img.unsqueeze(0)).squeeze(0).squeeze(0)
        binary_mask = to_binary_mask(output, dshape)
        cv2.imshow('ori', img_ori.copy())
        cv2.imshow('test', (binary_mask[..., np.newaxis] * img_ori.copy()).astype(np.uint8))
        cv2.waitKey()
        print(binary_mask)

if __name__=='__main__':
    import sys
    from cvtools import cv_load_image, clamp
    wts = sys.argv[1]
    pic = sys.argv[2]
    I = Infer(wts)
    if pic.startswith('http'):
        img = cv_load_image(pic)
        h, w, _ = img.shape
        if w > 1280:
            r, l, img = clamp(img)
    else:
        img = cv2.imread(pic)
    I.eval(img)
