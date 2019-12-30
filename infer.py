import torch
import numpy as np
from generalized_fcn import GeneralizedFCN
import cv2
from transforms import build_transforms

def to_binary_mask(output, dsize, thresh=0.5):
    binary_mask = output.cpu() > thresh
    binary_mask = binary_mask.float().numpy()
    binary_mask = cv2.resize(binary_mask, dsize=dsize, interpolation=cv2.INTER_LINEAR)
    return binary_mask
 
class Infer(object):
    def __init__(self, model_path):
        super(Infer, self).__init__()
        model = GeneralizedFCN()
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)
        model.eval()
        
        self.model = model
        self.transforms = build_transforms()

    def eval(self, impath):
        img_ori = cv2.imread(impath)
        dshape = (img_ori.shape[1], img_ori.shape[0])
        img = img_ori.copy()
        img = self.transforms(img)
        output = self.model(img.unsqueeze(0))
        binary_mask = to_binary_mask(output, dshape)
        cv2.imshow('ori', img_ori.copy())
        cv2.imshow('test', binary_mask[..., np.newaxis] * img_ori.copy())
        cv2.waitKey()
        print(binary_mask)

if __name__=='__main__':
    I = Infer('model_epoch_0.pth')
    I.eval('/data/data/jiahuan/localized_images_background//1310960.png')
