import warnings
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
from fcn_benchmark.utils.optimizer import OptimizerScheduler
from fcn_benchmark.utils.logger import Logger
from fcn_benchmark.utils.comm import synchronize, get_world_size, get_rank, is_main_process

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        torch.distributed.reduce(all_losses, dst=0)
        if torch.distributed.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses

def train_net(
              cfg,
              ngpus_per_node,
              model, 
              data_loader,
              optimizer,
              logg,
              ):
    model.train()
    start_training_time = time.time()
    end = time.time()
    
    for i in range(cfg.SOLVER.EPOCH):
        epoch_iter = len(data_loader)
        for iteration, (images, targets, _) in enumerate(data_loader):
            data_time = time.time() - end
            total_iter = i * epoch_iter + iteration
            if cfg.GPU is not None:
                images = images.cuda(cfg.GPU, non_blocking=True)
            targets = targets.cuda(cfg.GPU, non_blocking=True)
            
            loss_dict = model(images, targets) 
            losses = sum(loss for loss in loss_dict.values())
 
            optimizer.zero_grad()
            losses.backward()
            optimizer.step(total_iter)
            
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            batch_time = time.time() - end       
            end = time.time()
            logg.log(
                    total_iter, 
                    loss=losses_reduced.item(), 
                    lr=optimizer.lr,
                    data_time=data_time,
                    batch_time=batch_time,
                    )

            if total_iter % logg.interval == 0 and torch.distributed.get_rank() == 0:
                eta = (epoch_iter * cfg.SOLVER.EPOCH - total_iter) * logg.get('batch_time').global_avg
                eta_string = str(datetime.timedelta(seconds=int(eta)))
                info = {
                        'iter': total_iter, 
                        'loss': '{:.8f}'.format(logg.get('loss').median), 
                        'lr': '{:.4f}'.format(logg.get('lr').median), 
                        'data': '{:.4f}'.format(logg.get('data_time').median), 
                        'eta': eta_string, 
                        'mem': '{} MiB'.format(int(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0))
                        }
                #sys.stdout.write(str(info)+'\n')
                print(info)

        if torch.distributed.get_rank() == 0:
            torch.save(model.state_dict(), cfg.OUTPUT+'/model_epoch_{}.pth'.format(i))

def train_worker(gpu, ngpus_per_node, distributed, cfg):

    logg = Logger()
    logg.add('loss')
    logg.add('lr')
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

    #val_num = int(cfg.DATA.VAL_RATIO * len(data_set))
    #train_num = len(data_set) - val_num
    #train_dataset, val_dataset = random_split(data_set, [train_num, val_num])
    train_dataset = data_set
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
    from fcn_benchmark.config.defaults import _C 
    import argparse
    
    parser = argparse.ArgumentParser(description="training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args() 

    cfg = _C
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    train(cfg)

