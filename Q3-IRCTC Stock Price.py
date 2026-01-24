import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

#Reading The Excel Sheet

df=pd.read_excel("Data.xlsx",sheet_name="IRCTC Stock Price" )

#Calculating the mean and variance by pacakage

mean=np.mean(df["Price"])
variance=np.var(df["Price"])
print("Mean Results By Package:",mean)
print("Variance Results By Package:",variance)

#Calculating the mean and the variance by the function

def mean_function(arr):
    sum=0
    n=len(arr)
    for i in range(n):
        sum+=arr[i]
    average=sum/n
    return average
avg=mean_function(df["Price"])
def var_function(arr1):
    n=len(arr1)
    total=0
    variance1=0
    for i in range(n):
        variance1+=(arr1[i]-mean)**2
    variance2=variance1/n
    return variance2
variance2=var_function(df["Price"])

#Function that checks the computational complexity 
def avg_time(arr,func,run=10):
    times=[]
    for _ in range(run):
       start=time.time()
       func(arr)
       end=time.time()
       times.append(end-start)
    return sum(times)/run

t_np_mean=avg_time(df["Price"],np.mean)
t_np_variance=avg_time(df["Price"],np.var)
t_my_mean=avg_time(df["Price"],mean_function)
t_my_variance=avg_time(df["Price"],var_function)
print("The time calculated with numpy mean",t_np_mean)
print("The time calculated with numpy variance",t_np_variance)
print("Time form own mean",t_my_mean)
print("Time from own variance",t_my_variance)
wednesday_data=df[df["Day"]=="Wed"]
wednesday_mean=np.mean(wednesday_data["Price"])
apr_data=df[df["Month"]=="Apr"]
apr_mean=np.mean(apr_data["Price"])
total_days=len(df)
loss_days=df["Chg%"].apply(lambda x:x<0).sum()
prob_loss=loss_days/total_days
print("Probabilityof loss",prob_loss)
Wed_prof_days=wednesday_data["Chg%"].apply(lambda x:x>0).sum()
prob_profit_Wed=Wed_prof_days/(total_days)
conditional_probability=Wed_prof_days/len(wednesday_data)

print("Mean Results By Function:",avg)
print("Variance Results By Function:",variance2) 
print("Comparision for mean:",abs(mean-avg))
print("Comparision for Varinace:",abs(variance-variance2))    
print("Wednesday_mean:",wednesday_mean)
print("Comparision of wednesday and Total Mean:",abs(mean-wednesday_mean))
print("April Mean:",apr_mean)
print("Comparision of April and Total Mean:",abs(mean-apr_mean))
print("Probability of loss:",prob_loss)
print("Probability of profit on Wednesday:",prob_profit_Wed)
print("Condtional Probability:",conditional_probability)

plt.scatter(df["Day"], df["Chg%"])
plt.xlabel("Day of the Week")
plt.ylabel("Chg%")
plt.title("Chg% vs Day of the Week")
plt.show()
