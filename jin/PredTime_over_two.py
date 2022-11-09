import time
import json as json
import copy

import VGGD as VGGD
import GetComb
import Partition
from ClassPart import *


def get_layer_for_time_dict(path_to_json: str) -> dict:

    with open(path_to_json) as for_time:
        for_time_dict = json.load(for_time)
    
    return for_time_dict

def get_layer_back_time_dict(path_to_json: str) -> dict:

    with open(path_to_json) as back_time:
        back_time_dict = json.load(back_time)
    
    return back_time_dict

def get_time(num_procs, num_split,
             for_compute_time_lst,
             back_compute_time_lst,
             for_send_time_lst,
             back_send_time_lst):

    part_time_lst = [PartTime(for_compute_time=for_compute_time_lst[i],
                                back_compute_time=back_compute_time_lst[i],
                                for_send_time=for_send_time_lst[i],
                                back_send_time=back_send_time_lst[i],
                                num_split=num_split
                                ) for i in range(num_procs)]

    pred_class = PredTime(num_procs, num_split, part_time_lst)
    pred_class.pred_for_time()
    pred_class.pred_back_time()
    end_time = pred_class.get_end_time()

    return end_time

def get_dict_from_json(path):
    with open(path) as info:
        dict = json.load(info)
    return dict

def get_comm_time(size):
    bw = 64_000_000
    result = size/bw
    return result

if __name__=="__main__":
    num_split=8
    num_procs=8

    seq_model = VGGD.vgg16bn_imagenet().model
    name_lst = Partition.get_layer_name_lst(seq_model)

    comb_part = GetComb.get_comb_of_nway(len(name_lst),num_procs)

    print(len(comb_part))
    time.sleep(20)
    for_dict = get_dict_from_json("./VGGD_for_prof_bsize16")
    back_dict = get_dict_from_json("./VGGD_back_prof_bsize16")

    output_size_dict = Partition.get_layer_output_size("./VGGD_imagenet_output_size.json")

    min=999999

    start=time.time()

    for part_num, balance in enumerate(comb_part):
        print(part_num)
        part_lst = [None for i in range(len(balance))]
        for j, num_layer in enumerate(balance):
            if j==0:
                part_lst[j]=name_lst[j:balance[j]+1]
            else:
                part_lst[j]=name_lst[sum(balance[0:j]):balance[j]+1]

        for_compute_time_lst=[0 for i in range(num_procs)]
        back_compute_time_lst=[0 for i in range(num_procs)]

        for i, part in enumerate(part_lst):
            for name in part:
                for_compute_time_lst[i]+=for_dict[name]
                back_compute_time_lst[i]+=back_dict[name]
        
        size_lst = [output_size_dict[name_lst[i]] for i in balance]

        for_send_size_lst=[0 for i in range(num_procs-1)]

        for i in range(len(balance)-1):
            if sum(balance[0:i+1])<45:
                for_send_size_lst[i]=4*16*size_lst[i][1]*size_lst[i][2]*size_lst[i][3]
            else:
                for_send_size_lst[i]=4*16*size_lst[i][1]
        
        for_send_time_lst=[]
        for i in range(len(for_send_size_lst)):
            for_send_time_lst.append(get_comm_time(for_send_size_lst[i])*num_procs)
        
        
        back_send_time_lst=copy.deepcopy(for_send_time_lst)

        for_send_time_lst.append(0)
        back_send_time_lst.insert(0, 0)


        end_time = get_time(num_procs=num_procs,
        num_split=num_split,
        for_compute_time_lst=for_compute_time_lst,
        back_compute_time_lst=back_compute_time_lst,
        for_send_time_lst=for_send_time_lst,
        back_send_time_lst=back_send_time_lst)
        if min>end_time:
            min=end_time
            min_part=part_num
    
    end=time.time()
    total = end-start
    print(comb_part[min_part])
    print(total)