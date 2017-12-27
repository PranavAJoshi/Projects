
# coding: utf-8

# # Q1- To get the top 'n' words from the list of files. (Type-1)

# In[54]:


# Some necessary code
from collections import defaultdict, Counter
import pandas as pd
import re, datetime
from functools import partial
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt

def tokenize(message):
    message = message.lower()                       # convert to lowercase
    all_words = re.findall("[a-z0-9']+", message)   # extract the words
    return (all_words) 


# In[55]:


# Here, we are inserting code to create a map_reduce function that runs MapReduce on input using functions mapper and reducer
def map_reduce(inputs, mapper, reducer, n):
    collector = defaultdict(list)

    
    # Inserting a code to write a for loop over the inputs that calls mapper
    for i in inputs:
        for key, value in mapper(i):
            collector[key].append(value)

            
    # Inserting a code to write a return statement that calls the reducer
    return [output
           for key, values in collector.items()
           for output in reducer(key,values,n)]  

# Inserting a code to create a mapper function that return (file, value) for all files
def words_all_files_mapper(file_name):
    
    #Inserting a code to read all the files & then tokenize it.
    with open (file_name, "r") as f:
        for line in f:
            for word in tokenize(line):
                yield('All_files',(word,1))

                
# Inserting a code to create a reducer function that return the word with the highest total count    
def most_pop_top_n_word_reducer(file, words_and_counts, n):
    word_counts = Counter()
    
    
    # Inserting a code to write a for loop that will count all the words.
    for word, count in words_and_counts:
        word_counts[word] += count
    
        
    # Inserting code to find most common word and return that (key, value) pair   
    yield (file, word_counts.most_common(n))
    


# In[56]:


#Inserting the Files for which we want to check the top 'n' words.
file_names = ["Genesis.txt", 
            "Luke.txt",
            "Kings.txt"]


#Counting the top 12 words of all files. 
file_words = map_reduce(file_names,
                        words_all_files_mapper, 
                        most_pop_top_n_word_reducer, n=12)







#Inserting code to show the output in a format that looks good with the table.
for xy in file_words:
    print("\n Top " + str(len(file_words[0][1])) + " words in all the files are : ")
    print(tabulate(xy[1], headers = ["Words", "Counts"], tablefmt="fancy_grid")) 
    

#--------------------------------------------***-----------------------------------------------------------#


    #Inserting code to analyse the data with the help of bar plot for the top 'n' words b/w words & counts.   
    Table = xy[1]
    Label1=[]
    Count1=[]
    
    #To incraese the size of plot:-
    plt.figure(figsize=(16, 12)) 
    
    #Loop to get x & y i.e. Label(words) and its count.
    for i in range(0,len(Table)):
        Label1.append(Table[i][0])
        Count1.append(Table[i][1])

    y_pos = np.arange(len(Label1))
    
    
    #Inserting code to show the bar plot.
    plt.bar(y_pos, Count1)
    plt.xticks(y_pos, Label1)
    
    #Inserting code to label the x-axis & y-axis of bar plot.
    plt.xlabel('Words', fontsize=13)
    plt.ylabel('Total number of Counts', fontsize=13) 
    
    #Inserting code to write the title of bar plot.
    plt.title("Histogram between words & counts for the top " + str(len(xy[1])) + " words of All files \n", 
              fontsize=14, fontweight='bold')

    plt.show()


# # Q1 To get the top 'n' words of each file individually. (Type-2)

# In[57]:



# Inserting a code to create a mapper function that return (file, value) for each file update
def words_per_file_mapper(file_name):
    
    #Inserting a code to read all the files & then tokenize it.
    with open (file_name, "r") as f:
        for line in f:
            for word in tokenize(line):
                yield(file_name,(word,1))


                
                
# Q2 is done by 2 ways (In 1st type, we find the top 'n' words from the list of all files as asked.)
                     # (In 2nd type, we are finding top 'n' words of each file individually. )                             

    
# Here, in 2nd type we just changed the mapper.    


# In[58]:


#Inserting the Files for which we want to check the top 'n' words.
file_names = ["Genesis.txt", 
            "Luke.txt",
            "Kings.txt"]


#Counting the top *8* words per file. 
file_words = map_reduce(file_names,
                        words_per_file_mapper, 
                        most_pop_top_n_word_reducer, n=8)

#Inserting code to show the output in a format that looks good with the table.
for x in file_words:    
    print("\n Top " + str(len(x[1])) + " words in the file '" + x[0] + "' are :")  
    print(tabulate(x[1], headers = ["Words", "Counts"], tablefmt="fancy_grid"))  


# In[59]:


#Inserting code to analyse the data with the help of bar plot for the top 'n' words b/w words & counts.   
for x in file_words:
    Table = x[1]
    Label=[]
    Count=[]
    
    #To incraese the size of plot:-
    plt.figure(figsize=(16, 12)) 
    
    #Loop to get x & y i.e. Label(words) and its count.
    for i in range(0,len(Table)):
        Label.append(Table[i][0])
        Count.append(Table[i][1])

    y_pos = np.arange(len(Label))
    
    
    #Inserting code to show the bar plot.
    plt.bar(y_pos, Count)
    plt.xticks(y_pos, Label)
    
    #Inserting code to label the x-axis & y-axis of bar plot.
    plt.xlabel('Words', fontsize=13)
    plt.ylabel('Total number of Counts', fontsize=13) 
    
    #Inserting code to write the title of bar plot.
    plt.title("Histogram between words & counts for the top " + str(len(x[1])) + " words of file '" + x[0] + "'\n", 
              fontsize=14, fontweight='bold')

    plt.show()


# # Q2:- Here, we find top 'n' words from the list of files starting with the given letter(for e.g. here we are using letter 'N')   (Type-1)

# In[60]:


#Importing all the filenames listed in the directory.
import os             
all_files = os.listdir('C:\\Users\\pjoshi3\\mapreduce\\Untitled Folder')   # imagine you're one directory above test dir
print(all_files)


# In[69]:


# Here, we are inserting code to create a map_reduce function that runs MapReduce on input using functions mapper and reducer
def map_reduce1(inputs, mapper, reducer,n,a1a):
    """runs MapReduce on input using functions mapper and reducer"""
    collector = defaultdict(list)
    
    # write a for loop over the inputs that calls mapper
    for i in inputs:
        for key, value in mapper(i):
            collector[key].append(value)
    #print(collector)   
    # write a return statement that calls the reducer
    return [output
           for key, values in collector.items()
           for output in reducer(key,values,n,a1a)]  

# Inserting code to create a dictionary for the files available in directories from 'yob1880.txt' to 'yob2016.txt'
file_names=[]
for i in range(12,len(all_files)):
    file_names.append({"file_name" : "All_Files_1880to2016", 
                         "text"    :  all_files[i]})
    
# Inserting a code to create a mapper function that return (file, value) for each file update
def words_All_files_mapper1(file_name):
    """return (file, value) for each file update"""
    file = file_name["file_name"]
    with open (file_name["text"], "r") as f:
        for line in f:
            name,sex,num=line.split(",")
            for name in tokenize(line):
                yield(file,(name, int(num)))

# Inserting a code to create a reducer function that return the word with the total count    
def most_pop_top_n_word_reducer1(file, words_and_counts, n,a1a):
    """given a sequence of (word, count) pairs, 
    return the word with the highest total count"""
    word_counts = Counter()
    W_and_C=[]
    for i in range(0,len(words_and_counts)):
        if words_and_counts[i][0][0]==a1a:
            W_and_C.append(words_and_counts[i]) 
    for word, count in W_and_C:
        word_counts[word] += count
    
    # find most common word and return that (key, value) pair
    yield (file, word_counts.most_common(n))



# In[62]:


# Now, we are going to find top 5 values from the list of files(Year 1880 to 2016) starting with the letter "n"


# In[63]:


All_Files_words_1880_to_2016 = map_reduce1(file_names,
                        words_All_files_mapper1, 
                        most_pop_top_n_word_reducer1, 5, 'n')


#Inserting code to show the output in a format that looks good with the table.
for x in All_Files_words_1880_to_2016:    
    print("\n Top " + str(len(x[1])) + " words from the list of files starting with 1880 to 2016" + " are :")  
    print(tabulate(x[1], headers = ["Words", "Counts"], tablefmt="fancy_grid")) 
    
    
#--------------------------------------------***-----------------------------------------------------------#    
    
    #Inserting code to analyse the data with the help of bar plot for the top 'n' words b/w words & counts.   
    Table = x[1]
    Label1=[]
    Count1=[]
    
    #To incraese the size of plot:-
    plt.figure(figsize=(16, 12)) 
    
    #Loop to get x & y i.e. Label(words) and its count.
    for i in range(0,len(Table)):
        Label1.append(Table[i][0])
        Count1.append(Table[i][1])

    y_pos = np.arange(len(Label1))
    
    
    #Inserting code to show the bar plot.
    plt.bar(y_pos, Count1)
    plt.xticks(y_pos, Label1)
    
    #Inserting code to label the x-axis & y-axis of bar plot.
    plt.xlabel('\n Words', fontsize=13)
    plt.ylabel('\n Total number of Counts', fontsize=13) 
    
    #Inserting code to write the title of bar plot.
    plt.title("Histogram between words & counts for the top "+ str(len(x[1]))+ " words from the list of files \n 1880 to 2016", 
              fontsize=14, fontweight='bold')

    plt.show()


# # Q2:- Here, we find top 'n' words for each files individually(as mentioned) starting with any letter(for e.g. here we are using letter 'w')   (Type-2)

# In[64]:


def words_per_file_mapper2(file_name):
    """return (file, value) for each file update"""
    with open (file_name, "r") as f:
        for line in f:
            name,sex,num=line.split(",")
            for name in tokenize(line):
                yield(file_name,(name, int(num)))



# In[65]:


# Now, we are going to find top 4 values from each file individually(Year 1980 to 1982) starting with the letter "W"

# Here, we can give the file names for which we want to find top 'n' words and it will come individually.


# In[66]:


# Giving the files, for which we need top 4 words
file_names = ["yob1980.txt","yob1981.txt","yob1982.txt"]

file_words = map_reduce1(file_names,
                        words_per_file_mapper2, 
                        most_pop_top_n_word_reducer1,4,'w')


#--------------------------------------------***-----------------------------------------------------------#


#Inserting code to show the output in a format that looks good with the table.
for x in file_words:    
    print("\n Top " + str(len(x[1])) + " words in the file '" + x[0] + "' are :")  
    print(tabulate(x[1], headers = ["Words", "Counts"], tablefmt="fancy_grid")) 
    
    Table = x[1]
    Label=[]
    Count=[]
    
    #To incraese the size of plot:-
    plt.figure(figsize=(16, 12)) 
    
    #Loop to get x & y i.e. Label(words) and its count.
    for i in range(0,len(Table)):
        Label.append(Table[i][0])
        Count.append(Table[i][1])

    y_pos = np.arange(len(Label))
    
    
    #Inserting code to show the bar plot.
    plt.bar(y_pos, Count)
    plt.xticks(y_pos, Label)
    
    #Inserting code to label the x-axis & y-axis of bar plot.
    plt.xlabel('\n Words', fontsize=13)
    plt.ylabel('\n Total number of Counts', fontsize=13) 
    
    #Inserting code to write the title of bar plot.
    plt.title("Histogram between words & counts for the top " + str(len(x[1])) + " words of file '" + x[0] + "'\n", 
              fontsize=14, fontweight='bold')

    plt.show()   


# # Q3:- Here, we find top 'n' words from the list of files with the letter 'll' anywhere in name string.

# In[73]:


# Here, we are inserting code to create a map_reduce function that runs MapReduce on input using functions mapper and reducer   
def words_All_files_mapper3(file_name):
    """return (file, value) for each file update"""
    file = file_name["file_name"]
    with open (file_name["text"], "r") as f:
        for line in f:
            name,sex,num=line.split(",")
            for name in tokenize(line):
                yield(file,(name, int(num)))

                
# Inserting code to create a dictionary for the files available in directories from 'yob1880.txt' to 'yob2016.txt'
file_names=[]
for i in range(12,len(all_files)):
    file_names.append({"file_name" : "All_Files_1880to2016", 
                   "text" : all_files[i]})

# Inserting a code to create a reducer function that return the word with the total count  
def most_pop_top_n_word_reducer3(file, words_and_counts, n,a1a):
    """given a sequence of (word, count) pairs, 
    return the word with the highest total count"""
    word_counts = Counter()
    W_and_C=[]
    for i in range(0,len(words_and_counts)):
        if a1a in words_and_counts[i][0]:
            W_and_C.append(words_and_counts[i]) 
    for word, count in W_and_C:
        word_counts[word] += count
    
    # find most common word and return that (key, value) pair
    yield (file, word_counts.most_common(n))
    


# In[75]:


# Here, We are finding top 5 values with the letter 'll' in the names anywhere from the list of files(1880 to 2016).


# In[74]:


All_Files_words_1880_to_2016 = map_reduce1(file_names,
                        words_All_files_mapper3, 
                        most_pop_top_n_word_reducer3, 5, 'll')


#Inserting code to show the output in a format that looks good with the table.
for x in All_Files_words_1880_to_2016:    
    print("\n Top " + str(len(x[1])) + " words from the list of files 1880 to 2016 are :")  
    print(tabulate(x[1], headers = ["Words", "Counts"], tablefmt="fancy_grid")) 
    
    Table = x[1]
    Label=[]
    Count=[]
    
    #To incraese the size of plot:-
    plt.figure(figsize=(16, 12)) 
    
    #Loop to get x & y i.e. Label(words) and its count.
    for i in range(0,len(Table)):
        Label.append(Table[i][0])
        Count.append(Table[i][1])

    y_pos = np.arange(len(Label))
    
    
    #Inserting code to show the bar plot.
    plt.bar(y_pos, Count)
    plt.xticks(y_pos, Label)
    
    #Inserting code to label the x-axis & y-axis of bar plot.
    plt.xlabel('\n Words', fontsize=13)
    plt.ylabel('\n Total number of Counts', fontsize=13) 
    
    #Inserting code to write the title of bar plot.
    plt.title("Histogram between words & counts for the top "+ str(len(x[1]))+ " words from the list of files \n 1880 to 2016", 
              fontsize=14, fontweight='bold')

    plt.show()   


# In[80]:


# Here, We are finding top 15 values with the letter 'ppp' in the names anywhere from the list of files(1880 to 2016).


# In[79]:


All_Files_words_1880_to_2016 = map_reduce1(file_names,
                        words_All_files_mapper3, 
                        most_pop_top_n_word_reducer3,15,'pp')




#Inserting code to show the output in a format that looks good with the table.
for x in All_Files_words_1880_to_2016:    
    print("\n Top " + str(len(x[1])) + " words from the list of files 1880 to 2016 are :")  
    print(tabulate(x[1], headers = ["Words", "Counts"], tablefmt="fancy_grid")) 
    
    Table = x[1]
    Label=[]
    Count=[]
    
    #To incraese the size of plot:-
    plt.figure(figsize=(16, 12)) 
    
    #Loop to get x & y i.e. Label(words) and its count.
    for i in range(0,len(Table)):
        Label.append(Table[i][0])
        Count.append(Table[i][1])

    y_pos = np.arange(len(Label))
    
    
    #Inserting code to show the bar plot.
    plt.bar(y_pos, Count)
    plt.xticks(y_pos, Label)
    
    #Inserting code to label the x-axis & y-axis of bar plot.
    plt.xlabel('\n Words', fontsize=13)
    plt.ylabel('\n Total number of Counts', fontsize=13) 
    
    #Inserting code to write the title of bar plot.
    plt.title("Histogram between words & counts for the top "+ str(len(x[1]))+ " words from the list of files \n 1880 to 2016", 
              fontsize=14, fontweight='bold')

    plt.show()   

