import os
from itertools import permutations as permu

import torch as torch
import torch.multiprocessing as mp
from torch import distributed as dist

import GetComb
import distinit as distinit

def work(rank, A):
    B=[]

    for i, a in enumerate(A):
        print(rank, i)
        c_lst=[]
        for c in permu(a,8):
            c_c = list(c)
            if c_c not in c_lst:
                c_lst.append(c_c)
        for c in c_lst:
            B.append(c)
    
    with open("mp.txt"+str(rank), "w") as f:
        f.write(str(B))

    return B
  
if __name__ == '__main__':
    
    rank = int(os.environ["LOCAL_RANK"])
    distinit.setup(rank, 40)
    inputs = GetComb.get_eight()
    length = 17674
    
    if rank!=39:
        size = 442
        inputo = inputs[(rank)*size:(rank+1)*size+1]
    else:
        size = 436
        inputo = inputs[(rank)*442:(rank+1)*size+1]
        
    work(rank, inputo)
    
    