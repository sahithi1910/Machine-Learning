def summation(w,i):
    total=0
    for j in range(len(i)):
        total+=w(j)+i(j)
    return total 
def step(y):
    if y>=0:
        return 1
    else:
        return 0 
print(step(40)) 