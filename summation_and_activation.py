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
    def bipolar_step(y):
     if y>=0:
        return 1
     else:
        return -1
     def sigmoid(y):
        return (1/(1+math.exp(-y)))   
     def tan_h(y):
        return math.tan(y)
     def relu(y):
        return max(0,y)
     def leaky_relu(y,alpha):
        if y>=0:
           return y
        else:
           return y*alpha
               
print(step(40)) 
def comparator(target,actual):
   error=target-actual
   return error 
