import torch
import torchvision
import os
import requests
import copy
from cvtools import clamp, cv_load_image
from io import BytesIO
from PIL import Image
import numpy as np
from functools import wraps
from cvtorch.cvBox import BoxList
import pycocotools.mask as mask_utils
from segmentation_mask import SegmentationMask

def retry(func):
    @wraps(func)
    def wrapper(*args, **kw):
        times = 4
        while times >= 0:
            try:
                return func(*args, **kw)
            except Exception as e:
                times -= 1
                print('Retry.')
        print('Failed.')
    return wrapper

def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


class CocoDetection(torch.utils.data.Dataset):
    """
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, annFile, transform=None, target_transform=None, loader='OPENCV'):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = os.path.join(self.root, coco.loadImgs(img_id)[0]['file_name'])

        @retry
        def imloader(path, loader='OPNENCV'):
            assert loader in ('OPENCV', 'PIL'), 'Only support OPENCV, PIL for now.'
            if loader == 'OPENCV':
                return cv_load_image(path)
            else:
                if os.path.exists(path):
                    return Image.open(path).convert('RGB')
                else:
                    imfile = requests.get(path)
                    return Image.open(BytesIO(imfile.content)).convert('RGB')

        img = imloader(path, loader=self.loader)
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class COCODataset(CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None, clamp = False, loader='OPENCV'
    ):
        super(COCODataset, self).__init__(root, ann_file,loader=loader)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms
        self.clamp = clamp

        if clamp:
            for k in self.coco.imgs.keys():
                self.coco.imgs[k]['origin_width'] = self.coco.imgs[k]['width']
                self.coco.imgs[k]['width'] = self.coco.imgs[k]['clamp_width']
            for k in self.coco.anns.keys():
                im_id = self.coco.anns[k]['image_id']
                offset = self.coco.imgs[im_id]['off']
                if self.coco.anns[k] and 'segmentation' in self.coco.anns[k]:
                    for i, seg in enumerate(self.coco.anns[k]['segmentation']):
                        x = np.array(seg)
                        x[range(0, len(x), 2)] -= offset
                        self.coco.anns[k]['segmentation'][i] = x.tolist()

    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)
        anno = copy.deepcopy(anno)
        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]
        
        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        if self.clamp:
            imid = self.ids[idx]
            off = self.coco.imgs[imid]['off']
            r = self.coco.imgs[imid]['clamp_width'] + off
            if self.loader == 'PIL':
                img = img.crop((off, 0, r, img.size[1])) # Reference: https://pillow.readthedocs.io/en/stable/reference/Image.html
                wid, h = img.size
            else:
                img = img[:, off: r]
                h, wid, _c = img.shape
            boxes[:, 0] -= off
        else:
            if self.loader == 'PIL':
                wid, h = img.size
            else:
                h, wid, _c = img.shape
        target = BoxList(boxes, (wid, h), mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        if anno and "segmentation" in anno[0]:
            masks = [obj["segmentation"] for obj in anno]
            masks = SegmentationMask(masks, (wid, h))
            #masks = [mask_utils.decode(mask_utils.frPyObjects(m, h, wid)) for m in masks]
            target.add_field("masks", masks)
        #print(len(target))
        #print(target.bbox)
        #target = target.clip_to_image(remove_empty=True)
        #print(len(target))
        if self._transforms is not None:
            img, target = self._transforms(img, target)
            #self.show_image(img, target)
        #print(len(target))
        target_m = []
        if len(target.get_field('masks')) > 1:
            print(anno[0]['image_id'])
        for m in target.get_field('masks'):
            target_m.append(m.get_mask_tensor().float().unsqueeze(0))
        masks_target = torch.cat(target_m, dim=0)
        masks_target = (torch.sum(masks_target, dim=0) > 0).float().unsqueeze(0)
        return img, masks_target, idx


if __name__=='__main__':
    import cv2
    import sys
    from transforms import build_transforms
    df = '/core1/data/home/niuwenhao/workspace/data/detection/door_all_new.json'
    cd = COCODataset(df, '', True, transforms=build_transforms(),clamp=False)
    for im, targets, idx in cd:
        sys.stdout.write('{} / {}\r'.format(idx, len(cd)))
        sys.stdout.flush()
        continue
        print(im.shape, targets.get_field('masks').get_mask_tensor().shape)
        for m in targets.get_field('masks'):
            show = 255 * im * m.get_mask_tensor()
            show = show.numpy().astype(np.uint8)
            print(show)
            print(show.shape)
            print(show.dtype)
            cv2.imshow('test', show)
            cv2.waitKey()
