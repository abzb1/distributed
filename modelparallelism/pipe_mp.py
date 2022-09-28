import sys
import os
import time
import argparse

import torch as torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
#from torch.utils.tensorboard import SummaryWriter

import distinit as distinit
import VGGD as VGGD
import VGGE as VGGE
import dataloader as loader
import partition as partition
import trainloop as trainloop

parser = argparse.ArgumentParser(description="ModelParallelism")

parser.add_argument("--model", default="VGGD", type=str, metavar="model",
                    help="model in VGGD, VGGE (default: VGGD)")
parser.add_argument("--dataset", default="imagenet", type=str, metavar="dataset",
                    help="choose dataset in imagenet, cifar10 (dafault: imagenet)")                    
parser.add_argument("-e", "--epochs", default=1, type=int, metavar="N")
parser.add_argument("-b", "--batch-size", default=128, type=int, metavar="N")
parser.add_argument("-l", "--learning-rate", default=0.01, type=float, metavar="LR",
                    help="initial learning rate (default: 0.01)", dest="lr")
parser.add_argument("-m", "--momentum", default=0.9, type=float,
                    metavar="M", help="momentum")
parser.add_argument("-c", "--weight-decay", default=5e-4, type=float,
                    metavar="W", help="weight decay (default: 5e-4)",
                    dest="weight_decay")
parser.add_argument("-w", "--world-size", default=2, type=int, metavar="N",
                    help="number of processes for distributed training")
parser.add_argument("-t", "--num-split", default=4, type=int, metavar="N",
                    help="number of microbatch (default: 4)")
parser.add_argument("-a", "--master-address", default="Summit", type=str,
                    help="master address to set up distributed training")
parser.add_argument("-s", "--seed", default=None, type=int,
                    help="seed for initializing training.")
parser.add_argument("-p", "--print-freq", default=100, type=int, metavar="N",
                    help="print frequency (default: 100)")
parser.add_argument("-f", "--prof", default=False, type=bool, metavar="T/F",
                    help="Turn on PyTorch profiler option (default: False)")
parser.add_argument("--balance", default=[24, 29], type=list,
                    metavar="[N, N,]", help="balance to partition the model")

args = parser.parse_args()

if __name__ == "__main__":

    rank = int(os.environ["LOCAL_RANK"])
    distinit.setup(rank, args.world_size)

    gpu_num = torch.cuda.device_count()
    gpu = rank % gpu_num
    torch.cuda.set_device(gpu)

    if args.model == "VGGD":

        if args.dataset == "imagenet":
            seq_model = VGGD.vgg16bn_imagenet().model
            path = "VGGD_imagenet_output_size.json"
            train_dataloader, _ = loader.imageNetdata(args.batch_size)

        elif args.dataset == "cifar10":
            seq_model = VGGD.vgg16bn_cifar10().model
            path = "VGGD_cifar10_output_size.json"
            train_dataloader, _ = loader.cifar10data(args.batch_size)
    
    elif args.model == "VGGE":

        if args.dataset == "imagenet":
            seq_model = VGGE.vgg19bn_imagenet().model
            path = "VGGD_imagenet_output_size.json"
            train_dataloader, _ = loader.imageNetdata(args.batch_size)

        elif args.dataset == "cifar10":
            seq_model = VGGE.vgg19bn_cifar10().model
            path = "VGGD_cifar10_output_size.json"
            train_dataloader, _ = loader.cifar10data(args.batch_size)

    model_output_size_dict = partition.get_layer_output_size(path)

    layer_name_lst = partition.get_layer_name_lst(seq_model)

    partition.verify_model(seq_model, args.balance, args.world_size)
    model = partition.partition_model(rank, seq_model, args.balance, layer_name_lst)
    model.cuda()
    optimizer = optim.SGD(model.parameters(), args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    split_size = args.batch_size//args.num_split

    if rank == 0:
        _, output_size = partition.get_buff_size(rank,
                                                 model,
                                                 args.balance,
                                                 args.batch_size,
                                                 model_output_size_dict,
                                                 layer_name_lst)
        trainloop.mp_train_loop_first_prof(rank,
                                      model,
                                      train_dataloader,
                                      optimizer, scheduler,
                                      args.epochs,
                                      split_size,
                                      args.num_split,
                                      output_size,
                                      args.batch_size,
                                      args.world_size)

    elif rank == len(args.balance)-1:
        loss_fn = torch.nn.CrossEntropyLoss()
        input_size, _ = partition.get_buff_size(rank,
                                                model,
                                                args.balance,
                                                args.batch_size,
                                                model_output_size_dict,
                                                layer_name_lst)
        trainloop.mp_train_loop_last_prof(rank,
                                     model,
                                     loss_fn,
                                     train_dataloader,
                                     optimizer,
                                     scheduler,
                                     args.epochs,
                                     split_size,
                                     args.num_split,
                                     input_size,
                                     args.batch_size,
                                     args.world_size)

    else:
        input_size, output_size = partition.get_buff_size(rank,
                                                          model,
                                                          args.balance,
                                                          args.batch_size,
                                                          model_output_size_dict,
                                                          layer_name_lst)
        trainloop.mp_train_loop_middle_prof(rank,
                                       model,
                                       train_dataloader,
                                       optimizer,
                                       scheduler,
                                       args.epochs,
                                       split_size,
                                       args.num_split,
                                       input_size,
                                       output_size,
                                       args.batch_size,
                                       args.world_size)