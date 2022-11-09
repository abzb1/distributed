import time
import argparse
import json
import os
from collections import OrderedDict as OrderedDict

import torch as torch

import VGGDrelu_false as VGGD
import Partition

parser = argparse.ArgumentParser(description="PPProf")

parser.add_argument("--batch-size", default=16, type=int, metavar="N",
                    help="batch size to profile train")

args = parser.parse_args()

if __name__=="__main__":

    seq_model = VGGD.vgg16bn_imagenet().model
    name_lst = Partition.get_layer_name_lst(seq_model)

    torch.cuda.set_device(6)

    dummy_input = torch.rand(args.batch_size, 3, 224, 224, requires_grad=True).cuda()
    
    for_time_dict = OrderedDict()
    back_time_dict = OrderedDict()

    for n in range(2):
        
        input = dummy_input.clone()

        for i, layer in enumerate(seq_model):
            print(i)

            layer.cuda()
            layer.train()

            # forward
            torch.cuda.synchronize()
            start = time.time()
            output = layer(input)
            torch.cuda.synchronize()
            end = time.time()
            total = end-start
            if n==1:
                for_time_dict[name_lst[i]] = total
            
            # backward
            dummy_grad = torch.rand(output.size()).cuda()
            torch.cuda.synchronize()
            start = time.time()
            output.backward(dummy_grad)
            torch.cuda.synchronize()
            end = time.time()
            total = end-start
            if n==1:
                back_time_dict[name_lst[i]] = total

            # prepare for next
            input = output.clone().detach().requires_grad_(True)
    
    prof_for_dict = {}
    prof_back_dict = {}

    for name in name_lst:
        forms = for_time_dict[name]*1000
        backms = back_time_dict[name]*1000
        print(f"{name:>25s}  forward: {forms:>10f}ms")
        prof_for_dict[name]=forms
        prof_back_dict[name]=backms
        print(f"{name:>25s} backward: {backms:>10f}ms")

    
    for_prof_json_name="VGGD_for_prof_bsize"+str(args.batch_size)
    back_prof_json_name="VGGD_back_prof_bsize"+str(args.batch_size)
    for_path = "./"+for_prof_json_name
    back_path = "./"+back_prof_json_name

    with open(for_path, 'w') as outfile:
        json.dump(prof_for_dict, outfile)
    
    with open(back_path, 'w') as outfile:
        json.dump(prof_back_dict, outfile)