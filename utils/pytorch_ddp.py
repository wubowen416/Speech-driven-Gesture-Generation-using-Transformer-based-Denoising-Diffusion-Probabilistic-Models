import os
import time
from torch.distributed import init_process_group


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    addr = "localhost"
    port = 12357
    os.environ["MASTER_ADDR"] = addr
    os.environ["MASTER_PORT"] = str(port)
    while True:
        try:
            init_process_group(backend="nccl", rank=rank, world_size=world_size)
        except RuntimeError as e:
            if "The server socket has failed to listen on any local network address." in str(e):
                # change port and re-init
                port += 1
                os.environ["MASTER_PORT"] = str(port)
                print(f"[Warning] DDP try for new port number {port}.")
        else:
            print("[Info] DDP succesfully initilized.")
            break