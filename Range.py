def find_range(list1):
    length=len(list1)
    if length<3:
        return "Range Determination not possible"
    max1=max(list1)
    min1=min(list1)
    range1=max1-min1
    return range1
user_input=input("Enter numbers with space")
list1=list(map(int,user_input.split()))
result=find_range(list1)
print(result)
    
