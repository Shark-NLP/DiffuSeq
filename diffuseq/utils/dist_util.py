"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf

import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.

def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    backend = "gloo" if not th.cuda.is_available() else "nccl"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())

    if os.environ.get("LOCAL_RANK") is None:
        os.environ["MASTER_ADDR"] = hostname
        os.environ["RANK"] = str(0)
        os.environ["WORLD_SIZE"] = str(1)
        port = _find_free_port()
        os.environ["MASTER_PORT"] = str(port)
        os.environ['LOCAL_RANK'] = str(0)
    
    dist.init_process_group(backend=backend, init_method="env://")

    if th.cuda.is_available():  # This clears remaining caches in GPU 0
        th.cuda.set_device(dev())
        th.cuda.empty_cache()


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda:{os.environ['LOCAL_RANK']}")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file.
    """
    # if int(os.environ['LOCAL_RANK']) == 0:
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
