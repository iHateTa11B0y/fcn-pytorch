import warnings
import json
import os
import sys
import logging
import time
import datetime
from fcn_benchmark.data.dataset.coco import COCODataset
import torch
from fcn_benchmark.modeling.generalized_fcn import GeneralizedFCN
from torch import optim
import torch.backends.cudnn as cudnn
from fcn_benchmark.data.transforms import build_transforms
from torch.utils.data import DataLoader, random_split
from fcn_benchmark.utils.logger import Logger
from fcn_benchmark.utils.comm import synchronize, get_world_size, get_rank, is_main_process
import pycocotools.mask as mask_utils
from fcn_benchmark.utils.util import to_binary_mask, crop_weight    
import numpy as np

def test_net(
              cfg,
              ngpus_per_node,
              model, 
              data_loader,
              logg,
              ):
    model.eval()
    start_test_time = time.time()
    end = time.time()
    
    res_info = []
    
    for iteration, (images, targets, _) in enumerate(data_loader):
        data_time = time.time() - end
        if cfg.GPU is not None:
            images = images.cuda(cfg.GPU, non_blocking=True)
        targets = targets.cuda(cfg.GPU, non_blocking=True)
        
        res = model(images, targets)
        img_size, imid = (_[0][0].item(), _[0][1].item()), _[1]
        binary_mask = to_binary_mask(res.squeeze(0).squeeze(0), img_size)
        binary_mask = np.asfortranarray(binary_mask)
        rle = mask_utils.encode(binary_mask)
        res_info.append({'image_id': imid.item(), 'segmentation': rle, 'category_id': 1, 'score': 1.})

        batch_time = time.time() - end       
        end = time.time()
        logg.log(
                iteration, 
                data_time=data_time,
                batch_time=batch_time,
                )

        if iteration % logg.interval == 0 and torch.distributed.get_rank() == 0:
            eta = (len(data_loader) - iteration) * logg.get('batch_time').global_avg
            eta_string = str(datetime.timedelta(seconds=int(eta)))
            info = {
                    'iter': iteration, 
                    'data': '{:.4f}'.format(logg.get('data_time').median), 
                    'eta': eta_string, 
                    'mem': '{} MiB'.format(int(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0))
                    }
            #sys.stdout.write(str(info)+'\n')
            print(info)
    synchronize()
    torch.save(res_info, cfg.OUTPUT+'/segm.pth')
    #with open(cfg.OUTPUT+'/segm.json', 'w') as f:
    #    json.dump(res_info, f)


def test_worker(gpu, ngpus_per_node, distributed, cfg):

    logg = Logger()
    logg.add('data_time')
    logg.add('batch_time') 

    cfg.GPU = gpu

    if cfg.GPU is not ():
        print("Use GPU: {} for training".format(cfg.GPU))

    if distributed:
        if cfg.DIST_URL == "env://" and cfg.RANK == -1:
            cfg.RANK = int(os.environ["RANK"])
        if cfg.MULTIPROCESSING_DISTRIBUTED:
            cfg.RANK = cfg.RANK * ngpus_per_node + gpu
        torch.distributed.init_process_group(
                backend=cfg.DIST_BACKEND, 
                init_method=cfg.DIST_URL,
                world_size=cfg.WORLD_SIZE, 
                rank=cfg.RANK)
        synchronize()

    model = GeneralizedFCN()
    checkpoint = crop_weight(cfg.TEST_WEIGHT)
    model.load_state_dict(checkpoint)

    if distributed:
        if cfg.GPU is not ():
            torch.cuda.set_device(cfg.GPU)
            model.cuda(cfg.GPU)
            cfg.SOLVER.BATCH_SIZE = int(cfg.SOLVER.BATCH_SIZE / ngpus_per_node)
            cfg.WORKERS = int(cfg.WORKERS / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.GPU])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif cfg.GPU is not ():
        torch.cuda.set_device(cfg.GPU)
        model = model.cuda(cfg.GPU)
    else:
        model = torch.nn.DataParallel(model).cuda()

    test_dataset = COCODataset(cfg.DATA.TESTSET, 
                           '', 
                           True, 
                           transforms=build_transforms(), 
                           clamp=False)

    test_dataloader = DataLoader(test_dataset, 
                                  batch_size=cfg.SOLVER.BATCH_SIZE, 
                                  shuffle=False, 
                                  num_workers=cfg.WORKERS, 
                                  pin_memory=True,
                                  )

    test_net(cfg, 
              ngpus_per_node, 
              model, 
              test_dataloader, 
              logg)

def test(cfg):

    if cfg.GPU is not ():
        warnings.warn('You have chosen a specific GPU. This will completely '
                'disable data parallelism.')

    if cfg.DIST_URL == "env://" and cfg.WORLD_SIZE == -1:
        cfg.WORLD_SIZE = int(os.environ["WORLD_SIZE"])

    distributed = cfg.WORLD_SIZE > 1 or cfg.MULTIPROCESSING_DISTRIBUTED

    ngpus_per_node = torch.cuda.device_count()
    
    if cfg.MULTIPROCESSING_DISTRIBUTED:
        cfg.WORLD_SIZE = ngpus_per_node * cfg.WORLD_SIZE
        torch.multiprocessing.spawn(test_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, distributed, cfg))
    else:
        test_worker(cfg.GPU, distributed, cfg)


if __name__=='__main__':
    from yacs.config import CfgNode as CN
    _C = CN()
    _C.GPU = ()
    _C.DIST_URL = "env://" #'tcp://224.66.41.62:23456'
    _C.WORLD_SIZE = -1
    _C.MULTIPROCESSING_DISTRIBUTED = True
    _C.RANK = -1
    _C.DIST_BACKEND = 'nccl'
    _C.WORKERS = 8
    _C.OUTPUT = 'weights'
    _C.TEST_WEIGHT = '/home/niuwenhao/Repo/fcns/weights/model_epoch_9.pth'

    _C.DATA = CN()
    _C.DATA.TRAINSET = "/core1/data/home/niuwenhao/workspace/data/detection/door_all_new.json"
    _C.DATA.TESTSET = "/core1/data/home/niuwenhao/data/tiny_bg.json"

    _C.SOLVER = CN()
    _C.SOLVER.LR = 0.02
    _C.SOLVER.BATCH_SIZE = 4
    _C.SOLVER.STEPS = (64000, 80000)
    _C.SOLVER.EPOCH = 12
    _C.SOLVER.MOMENTUM = 0.9
    _C.SOLVER.WEIGHT_DECAY = 1e-4

    cfg = _C

    test(cfg)
        
