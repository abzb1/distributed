from itertools import permutations as permu

A = [1,1,2,3]

for b in permu(A, 4):
    print(b)