input_string=input("Enter the String")
freq={}
def highest_count(input_string):
    for character in input_string:
        if character in freq:
           freq[character]+=1
        else:
            freq[character]=1
        occurence_character=None
        occurence_count=0
    for character in freq:
        if freq[character]>occurence_count:
            occurence_count=freq[character]
            occurence_character=character
    return occurence_count,occurence_character
occurence_count,occurence_character=highest_count(input_string)
print("occurence_character:",occurence_character)
print("occurence_count:",occurence_count)
