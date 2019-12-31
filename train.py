import warnings
import os
from coco import COCODataset
import torch
from generalized_fcn import GeneralizedFCN
from torch import optim
import torch.backends.cudnn as cudnn
from transforms import build_transforms
from torch.utils.data import DataLoader, random_split
from optimizer import OptimizerScheduler
from logger import Logger


def train_net(
              cfg,
              ngpus_per_node,
              model, 
              data_loader,
              optimizer,
              logg,
              ):
    model.train()
    for i in range(cfg.SOLVER.EPOCH):
        epoch_iter = len(data_loader)
        for iteration, (images, targets, _) in enumerate(data_loader):
            total_iter = i * epoch_iter + iteration
            if cfg.GPU is not None:
                images = images.cuda(cfg.GPU, non_blocking=True)
            targets = targets.cuda(cfg.GPU, non_blocking=True)
            
            loss = model(images, targets) 
 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(total_iter)
            
            logg.log(total_iter, loss=loss.item(), lr=optimizer.lr)

            if iteration % 10 == 0:
                flag = not cfg.MULTIPROCESSING_DISTRIBUTED or (cfg.MULTIPROCESSING_DISTRIBUTED and cfg.RANK % ngpus_per_node == 0)
                logg.wait(flag, flush=True)

        if not cfg.MULTIPROCESSING_DISTRIBUTED or (cfg.MULTIPROCESSING_DISTRIBUTED and cfg.RANK % ngpus_per_node == 0):
            torch.save(model.state_dict(), 'model_epoch_{}.pth'.format(i))

def train_worker(gpu, ngpus_per_node, distributed, cfg):

    logg = Logger()
    logg.add('loss')
    logg.add('lr')

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


    model = GeneralizedFCN()

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


    data_set = COCODataset(cfg.DATA.TRAINSET, 
                           '', 
                           True, 
                           transforms=build_transforms(), 
                           clamp=False)

    cudnn.benchmark = True

    val_num = int(cfg.DATA.VAL_RATIO * len(data_set))
    train_num = len(data_set) - val_num
    train_dataset, val_dataset = random_split(data_set, [train_num, val_num])
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=cfg.SOLVER.BATCH_SIZE, 
                                  shuffle=(train_sampler is None), 
                                  num_workers=cfg.WORKERS, 
                                  pin_memory=True,
                                  sampler=train_sampler)

    optimizer = OptimizerScheduler(model, cfg)     
    train_net(cfg, 
              ngpus_per_node, 
              model, 
              train_dataloader, 
              optimizer, 
              logg)

def train(cfg):

    if cfg.GPU is not ():
        warnings.warn('You have chosen a specific GPU. This will completely '
                'disable data parallelism.')

    if cfg.DIST_URL == "env://" and cfg.WORLD_SIZE == -1:
        cfg.WORLD_SIZE = int(os.environ["WORLD_SIZE"])

    distributed = cfg.WORLD_SIZE > 1 or cfg.MULTIPROCESSING_DISTRIBUTED

    ngpus_per_node = torch.cuda.device_count()
    
    if cfg.MULTIPROCESSING_DISTRIBUTED:
        cfg.WORLD_SIZE = ngpus_per_node * cfg.WORLD_SIZE
        torch.multiprocessing.spawn(train_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, distributed, cfg))
    else:
        train_worker(cfg.GPU, distributed, cfg)


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

    _C.DATA = CN()
    _C.DATA.TRAINSET = "/core1/data/home/niuwenhao/data/tiny_bg.json"
    _C.DATA.VAL_RATIO = 0.1

    _C.SOLVER = CN()
    _C.SOLVER.LR = 0.1
    _C.SOLVER.BATCH_SIZE = 4
    _C.SOLVER.STEPS = (100, 200, 300)
    _C.SOLVER.EPOCH = 10
    _C.SOLVER.MOMENTUM = 0.9
    _C.SOLVER.WEIGHT_DECAY = 1e-4

    cfg = _C

    train(cfg)
        
