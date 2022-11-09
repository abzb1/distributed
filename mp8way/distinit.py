import os
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    print(f"setup {rank} th process")
    os.environ["MASTER_ADDR"] = "Summit"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["NCCL_NET_GDR_LEVEL"] = "4"
    os.environ["OMP_NUM_THREADS"] = str(int(mp.cpu_count() / world_size))
    dist.init_process_group("nccl", rank = rank, world_size = world_size)
    print(f"{rank} th process setup complete")