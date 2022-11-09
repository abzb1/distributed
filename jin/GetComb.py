import json

from itertools import combinations_with_replacement as H
from itertools import permutations as permu


def combelem2(lst):
    result = []
    for i in range(len(lst)):
        result.append([lst[i][1],lst[i][0]])
    return result

def DIVID(a, n):
    A = []
    for x in H(range(1, a), n):
        
        if all([g>a//n for g in x]):
            break
        
        if sum(x)==a:
            A.append(list(x))
        
    return A

def get_comb_of_two_way():
    result = DIVID(53, 2)
    revers = list(reversed(combelem2(result)))
    result = result+revers
    return result

def get_eight():
    with open("g.json") as eight:
        eight_dict = json.load(eight)
    return eight_dict["comb"]

def get_comb_of_nway(a,n):
    if n==8:
        with open("result.json") as t:
            res_dict = json.load(t)
            res = res_dict["comb"]
        return res
    else:
        A = DIVID(a, n)

    B=[]

    for i, a in enumerate(A):
        print(i)
        c_lst=[]
        for c in permu(a,n):
            c_c = list(c)
            if c_c not in c_lst:
                c_lst.append(c_c)
        for c in c_lst:
            B.append(c)
            
    f=open("result"+str(n)+".txt", "w")
    f.write(str(B))
    f.close()

    return B

get_comb_of_nway(53, 4)