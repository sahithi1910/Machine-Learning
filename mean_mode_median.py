import random
random_list=[]
for i in range(25):
    a=random.randint(1,10)
    random_list.append(a)
print(random_list)
def mean(random_list):
    sum_in_list=sum(random_list)
    len_of_list=len(random_list)
    mean=sum_in_list/(len_of_list)
    return mean
def median(random_list):
    random_list.sort()
    n=len(random_list)
    if(n%2==0):
        median=(random_list[n//2]+random_list[(n//2)-1])/2
        return median
    else:
        median=random_list[n//2]
        return median
def mode(random_list):
    freq={}
    for num in random_list:
        if num in freq:
            freq[num]+=1
        else:
            freq[num]=1
    mode=None
    max1=0
    for num in freq:
        if freq[num]>max1:
            max1=freq[num]
            mode=num
    return mode
print("Mean:",mean(random_list))
print("Median:",median(random_list))
print("Mode:",mode(random_list))
     
    
