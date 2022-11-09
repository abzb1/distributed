import time
import json

with open("result.json") as t:
    res_dict = json.load(t)
    res = res_dict["comb"]
    print(len(res))
