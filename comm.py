import torch

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not torch.distributed.is_available():
        return
    if not torch.distributed.is_initialized():
        return
    world_size = torch.distributed.get_world_size()
    if world_size == 1:
        return
    torch.distributed.barrier()


def get_world_size():
    if not torch.distributed.is_available():
        return 1
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()

def get_rank():
    if not torch.distributed.is_available():
        return 0
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()

def is_main_process():
    return get_rank() == 0
