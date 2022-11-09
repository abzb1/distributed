def processfile():
    with open("result.txt", "w") as r:
        r.write('{"comb":[')
        for i in range(40):
            print(i)
            with open("mp.txt"+str(i), "r") as f:
                temp = f.read()
                temp = temp.lstrip("[")
                temp = "["+temp
                temp = temp.rstrip("]")
                temp = temp+"]"
                r.write(temp)
            if i != 39:
                r.write(",")
        r.write(']}')

processfile()