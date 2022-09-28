import sys
import os
import time
import random
import argparse

import numpy as np

import torch as torch
from torch import distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.tensorboard import SummaryWriter

import VGGD
import distributeddataloader as ddl
import distinit as distinit


parser = argparse.ArgumentParser(description="VGG16 DataParallelism")

parser.add_argument("-e", "--epochs", default=1, type=int, metavar="N")
parser.add_argument("-b", "--batch-size", default=128, type=int, metavar="N")
parser.add_argument("-l", "--learning-rate", default=0.1, type=float, metavar="LR", help="initial learning rate", dest="lr")
parser.add_argument("-m", "--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument("-c", "--weight-decay", default=1e-4, type=float, metavar="W", help="weight decay (default: 1e-4)", dest="weight_decay")
parser.add_argument("-n", "--num-procs", default=2, type=int, help="number of processes for distributed training")
parser.add_argument("-a", "--master-address", default="Summit", type=str, help="master address to set up distributed training")
parser.add_argument("-s", "--seed", default=None, type=int, help="seed for initializing training.")
parser.add_argument("-p", "--print-freq", default=100, type=int, metavar="N", help="print frequency (default: 10)")

args = parser.parse_args()

def run_ddp():
    world_size = args.num_procs
    rank = int(os.environ["LOCAL_RANK"])
    
    if args.seed is not None:
        torch.backends.cudnn.benchmark = False
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(seed=args.seed + rank)
        random.seed(args.seed + rank)
        torch.backends.cudnn.deterministic = True

    distinit.setup(rank, world_size)

    torch.cuda.set_device(rank)
    model = VGGD.vgg16bn().cuda()
    ddp_model = DDP(model, device_ids=[rank], broadcast_buffers=True, bucket_cap_mb=25, gradient_as_bucket_view=True)

    print(f"Running DDP on rank {rank}")
    
    loss_fn = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(ddp_model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    num_workers = int(mp.cpu_count() / world_size)
    train_dataloader, test_dataloader = ddl.imageNetdata(args.batch_size, num_workers, args.epochs)
    size = len(train_dataloader.dataset)
    num_batches = len(train_dataloader)

    # writer = SummaryWriter("./runs/intra_ddp_torchrun_imageNet_vgg16D_batchsize_"+str(args.batch_size)+"_worldsize_"+str(world_size))
    running_loss = 0.0
    epochs = args.epochs * world_size
    for t in range(epochs):
        print(f"epoch : {t}\n")
        start=time.time()
        sum = 0.0
        #train
        with torch.profiler.profile(schedule=torch.profiler.\
                                             schedule(wait=100,\
                                                      warmup=100,\
                                                      active=10,\
                                                      repeat=1,\
                                                      skip_first=100),
                                    on_trace_ready = torch.profiler.\
                                                     tensorboard_trace_handler("./prof/intra_ddp_imageNet_VGGD_batchsize_"+\
                                                                               str(args.batch_size)+\
                                                                               "_worldsize_"+str(args.num_procs)),
                                    record_shapes = True,
                                    profile_memory = True,
                                    with_flops = True,
                                    with_stack = True
                                    ) as prof:        
            for batch, (X, y) in enumerate(train_dataloader):
                tic = time.time()
                model_in, label = X.cuda(), y.cuda()
                optimizer.zero_grad()
                pred = ddp_model(model_in)
                loss = loss_fn(pred, label)
                loss.backward()
                optimizer.step()
                prof.step()
                running_loss += loss.item()
                toc = time.time()
                sum += toc-tic

                if batch % args.print_freq == 0:
                    if batch == 0:
                        continue
                    # writer.add_scalar("training loss", running_loss / args.print_freq, t * len(train_dataloader) + batch)
                    running_loss = 0.0
                    loss, current = loss.item(), batch * args.batch_size
                    step_time = (sum/args.print_freq)*1000
                    print(f"{rank}, loss : {loss:>8f} [{current:>7d}/{size:>7d}] avgsteptime : {step_time:>10f} ms")
                    sum = 0.0
        end=time.time()
        total = end-start
        print(f"{t} th epoch end")
        print(f"time elapsed {total:>3f} s\n")

        scheduler.step()

    torch.save(model.state_dict(), "ddp_model_weights.pth")
    dist.destroy_process_group()

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    assert args.num_procs <= n_gpus, "too many procs!"
    run_ddp()
