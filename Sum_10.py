my_list=[2,7,4,1,3,6]
def know_Sum_is_10(num_in_list): # function to see what are the numbers that give the sum as 10
    length=len(num_in_list)
    count=0
    for i in range(length):
        for j in range(i):
            if(num_in_list[i]+num_in_list[j]==10):# We check that where we will get 10
                count+=1
    return count
result= know_Sum_is_10(my_list)
print(result)
