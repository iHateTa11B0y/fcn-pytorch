from cvtorch import cvTransforms as cvT
from cvtorch.cvFunctional import ColorJitter
import cv2

class Resize(object):
    def __init__(self, rsize):
        self.rsize = (rsize, rsize)
    def __call__(self, image, target=None):
        image = cv2.resize(image, dsize=self.rsize, interpolation=cv2.INTER_LINEAR)
        if target is not None:
            target = target.resize((image.shape[1], image.shape[0]))
        return image, target

def build_transforms(image_loader="OPENCV"):
    assert image_loader in ('OPENCV',), 'only support OPENCV, PIL image loader'
    
    brightness = 0.0
    contrast = 0.0
    saturation = 0.0
    hue = 0.0
    rsize = 572
    mean = [102.9801, 115.9465, 122.7717]
    std = [1., 1., 1.]

    normalize_transform = cvT.NormalizeAsTorch(mean, std)
    #color_jitter = cvT.ColorJitter(
    #    brightness=brightness,
    #    contrast=contrast,
    #    saturation=saturation,
    #    hue=hue,
    #)

    transforms = cvT.Compose(
	[
		#color_jitter,
		Resize(rsize),
		#cvT.RandomHorizontalFlip(0.5),
		#cvT.RandomVerticalFlip(0.5),
		cvT.ToTensor(),
		normalize_transform,
	]
    )
    return transforms
