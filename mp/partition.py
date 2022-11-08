import json as json
from collections import OrderedDict as OrderedDict

import torch as torch
from torch import nn as nn

import VGGD as VGGD

def verify_model(model, balance, world_size):

    if not isinstance(model, nn.Sequential):
        raise TypeError("module must be nn.Sequential to be partitioned")

    if sum(balance)!=len(model) or len(balance)!=world_size:
        raise ValueError("Need accurate balance!")

def get_layer_name_lst(model: nn.Sequential) -> list:

    name_lst = []

    for name, _ in model.named_children():
        name_lst.append(name)

    return name_lst

def get_layer_output_size(path_to_json: str) -> dict:

    with open(path_to_json) as output_info:
        output_size_dict = json.load(output_info)
    
    return output_size_dict

def get_buff_size(rank: int, model: nn.Sequential,
                  balance: list, batch_size: int,
                  output_size_dict: dict, name_lst: list) -> list:

    if rank != 0:    
        input_from_layer_num = sum(balance[:rank])-1
        input_size = output_size_dict[name_lst[input_from_layer_num]]
        input_size[0] = batch_size

    else:
        input_size = [batch_size, 3, 224, 224]

    output_layer_num = sum(balance[:rank+1])-1
    output_size = output_size_dict[name_lst[output_layer_num]]
    output_size[0] = batch_size

    return input_size, output_size

def partition_model(rank: int, model: nn.Sequential, balance: list, name_lst: list) -> nn.Sequential:

    start_layer_num = sum(balance[:rank])
    end_layer_num = sum(balance[:rank+1])

    layer_dict = OrderedDict()
     
    for i in range(start_layer_num, end_layer_num):
        layer_dict[name_lst[i]] = model[i]
    
    partitioned_model = nn.Sequential(layer_dict)

    return partitioned_model