import multiprocessing as mp
from itertools import permutations as permu

import GetComb

def work(A):
    B=[]

    for i, a in enumerate(A):
        print(i)
        c_lst=[]
        for c in permu(a,8):
            c_c = list(c)
            if c_c not in c_lst:
                c_lst.append(c_c)
        for c in c_lst:
            B.append(c)

    return B
  
if __name__ == '__main__':
  # Pool 객체 초기화
  pool = mp.Pool(processes=40)

  inputs = GetComb.get_eight()
  outputs = pool.map(work(inputs), inputs)
  
  print(outputs)