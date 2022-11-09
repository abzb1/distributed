import json as json

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
    bw = 64000000000
    result = size/bw
    return result

if __name__=="__main__":
    num_split=8
    num_procs=2

    seq_model = VGGD.vgg16bn_imagenet().model
    name_lst = Partition.get_layer_name_lst(seq_model)

    comb_part = GetComb.get_comb_of_two_way()

    for_dict = get_dict_from_json("./VGGD_for_prof_bsize16")
    back_dict = get_dict_from_json("./VGGD_back_prof_bsize16")

    output_size_dict = Partition.get_layer_output_size("./VGGD_imagenet_output_size.json")

    min=999999

    for i, balance in enumerate(comb_part):
        first_part=name_lst[0:balance[0]+1]
        second_part=name_lst[balance[0]:]

        for_compute_time_lst=[0 for i in range(num_procs)]
        back_compute_time_lst=[0 for i in range(num_split)]

        for name in first_part:
            for_compute_time_lst[0]+=for_dict[name]
        for name in second_part:
            for_compute_time_lst[1]+=for_dict[name]

        for name in first_part:
            back_compute_time_lst[0]+=back_dict[name]
        for name in second_part:
            back_compute_time_lst[1]+=back_dict[name]
        
        size = output_size_dict[name_lst[balance[0]]]

        if balance[0]<45:
            for_send_size=4*16*size[1]*size[2]*size[3]
        else:
            for_send_size=4*16*size[1]
        
        back_send_size = for_send_size

        for_send_time = get_comm_time(for_send_size)
        back_send_time=for_send_time

        for_send_time_lst=[for_send_time,0]
        back_send_time_lst=[0,back_send_time]


        end_time = get_time(num_procs=num_procs,
        num_split=num_split,
        for_compute_time_lst=for_compute_time_lst,
        back_compute_time_lst=back_compute_time_lst,
        for_send_time_lst=for_send_time_lst,
        back_send_time_lst=back_send_time_lst)
        if min>end_time:
            min=end_time
            part=balance
    
    print(part)