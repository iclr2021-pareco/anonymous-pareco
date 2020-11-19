# -*- coding:utf-8  -*-

import functools
import os
import socket
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from collections import OrderedDict


from torch._utils import _flatten_dense_tensors
from torch._utils import _unflatten_dense_tensors
from torch._utils import _take_tensors


gpu_id = None


def init_dist(distributed=True, backend='nccl'):
    global gpu_id

    if distributed:
        mp.set_start_method('spawn')
        if dist.is_initialized():
            return

        # for slurm
        if os.environ.get('SLURM_PROCID', None) is not None:
            rank, world_size, url = _slurm_init_distributed()

        # for environment variable
        else:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            url = None

        num_gpus = torch.cuda.device_count()
        gpu_id = rank % num_gpus
        torch.cuda.set_device(gpu_id)

        dist.init_process_group(backend, init_method=url, rank=rank, world_size=world_size)


def _slurm_init_distributed():
    def find_free_port():
        s = socket.socket()
        s.bind(('', 0))
        return s.getsockname()[1]

    rank = int(os.environ['SLURM_PROCID'])
    world_size = int(os.environ['SLURM_NPROCS'])
    job_id = os.environ['SLURM_JOBID']
    host_file = 'config/dist_url.' + job_id + '.txt'

    # for master
    if rank == 0:
        ip = socket.gethostbyname(socket.gethostname())
        port = find_free_port()
        dist_url = 'tcp://{}:{}'.format(ip, port)
        with open(host_file, 'w') as f:
            f.write(dist_url)

    else:
        while not os.path.exists(host_file):
            time.sleep(1)
        with open(host_file, 'r') as f:
            dist_url = f.read()

    return rank, world_size, dist_url


def get_rank():
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def get_world_size():
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def is_master():
    return get_rank() == 0


def master(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_master():
            return func(*args, **kwargs)
        else:
            return None
    return wrapper


def all_reduce(tensor, div=False):
    world_size = get_world_size()
    if world_size == 1:
        return tensor

    with torch.no_grad():
        dist.all_reduce(tensor)
        if div:
            tensor.div_(world_size)

    return tensor


def scaled_all_reduce(tensors, div=True):
    """Performs the scaled all_reduce operation on the provided tensors.
    The input tensors are modified in-place. Currently supports only the sum
    reduction operator. The reduced values are scaled by the inverse size of the
    process group (equivalent to cfg.NUM_GPUS).
    """
    # There is no need for reduction in the single-proc case
    world_size = get_world_size()
    if world_size == 1:
        return tensors
    # Queue the reductions
    reductions = []
    for tensor in tensors:
        reduction = torch.distributed.all_reduce(tensor, async_op=True)
        reductions.append(reduction)
    # Wait for reductions to finish
    for reduction in reductions:
        reduction.wait()
    # Scale the results
    if div:
        for tensor in tensors:
            tensor.mul_(1.0 / world_size)
    return tensors

def _allreduce_coalesced(tensors, world_size, bucket_size_mb=-1):
    if bucket_size_mb > 0:
        bucket_size_bytes = bucket_size_mb * 1024 * 1024
        buckets = _take_tensors(tensors, bucket_size_bytes)
    else:
        buckets = OrderedDict()
        for tensor in tensors:
            tp = tensor.type()
            if tp not in buckets:
                buckets[tp] = []
            buckets[tp].append(tensor)
        buckets = buckets.values()

    for bucket in buckets:
        flat_tensors = _flatten_dense_tensors(bucket)
        dist.all_reduce(flat_tensors)
        flat_tensors.div_(world_size)
        for tensor, synced in zip(
                bucket, _unflatten_dense_tensors(flat_tensors, bucket)):
            tensor.copy_(synced)

def allreduce_grads(model, coalesce=True, bucket_size_mb=-1):
    grads = [
        param.grad.data for param in model.parameters()
        if param.requires_grad and param.grad is not None
    ]
    world_size = get_world_size()
    if coalesce:
        _allreduce_coalesced(grads, world_size, bucket_size_mb)
    else:
        for tensor in grads:
            dist.all_reduce(tensor.div_(world_size))


def average_gradient(params):
    world_size = get_world_size()
    if world_size == 1:
        return

    for param in params:
        if param.requires_grad and param.grad is not None:
            dist.all_reduce(param.grad.data)


def barrier():
    world_size = get_world_size()
    if world_size > 1:
        dist.barrier()
