
# coding: utf-8

# # Summary

# In[2]:

# What we are doing in this question:-  
#  -> loading of iris data (all features).
#  -> spliting 80% of this data for training & remaining 20% data for testing.
#  -> Now, we creates a classifier, then we train & fit this classifier object, and finally, we predict our testing data.
#  -> Calculating accuracy, precision, recall, and f1 score & ploting this versus no. of neighbours.
#  -> To get a clear insight, we are plotting the performance measures with neighbors individually & together also.
#  -> Now, we plot classification region on the basis of Best & Worst classifier.

# Note:- We are doing this process:- 
#        1) for k values 1 to 50. Hence, no. of neighbours in X-axis of plot is from 1 to 50
#        2) for both uniform & distance weighted classifier.
#        3) for both micro & macro averaged precision, recall and f1 score.
#        4) calculating & plotting accuracy, precision, recall and f1 score
#           for both overall data(all classes included) and for each classes(class 1, class 2, class 3) individually.
#        5) plotting the classification region for both uniform & distance model (for the best & worst classifier.)
           


# In[1]:

# What we observed & concluded after completing this second question of the assignment:- 
#   -> For micro-averaged, precision, recall and f1 score are same, & therefore graph is also similar.
#   -> For macro-averaged, precision, recall and f1 are different, but, with a very small margin as we can see in the plots.
#   -> We can also calculate precision, recall and f1 for each classes(class 1, class 2, class 3) using confusion matrix.
   
#   -> Best classifier is that classifier which shows best accuracy or less error rate at any particular value of "k".
#   -> Worst classifier is that classifier which shows least accuracy or high error rate at any particular value of "k".
  


# # Question 2 :-   Type 1
# #                          Use PCA to reduce the iris data set to 2 components.  
# #                          Plot accuracy, precision, recall, and f1-score ("For all class together")  vs. the number of neighbors with a 20% test split on the iris data 
# #                          And, plot the classification region for best & worst classifier
# 

# In[1]:

# import the libraries
from collections import Counter
import math, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from matplotlib.colors import ListedColormap

from sklearn import neighbors, datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import seaborn as sns
sns.set()

def split_data(data, prob):
    """split data into fractions [prob, 1 - prob]"""
    results = [], []
    for row in data:
        results[0 if random.random() < prob else 1].append(row)
    return results

def train_test_split(x, y, test_pct):
    data = list(zip(x, y))                        # pair corresponding values
    train, test = split_data(data, 1 - test_pct)  # split the dataset of pairs
    x_train, y_train = list(zip(*train))          # magical un-zip trick
    x_test, y_test = list(zip(*test))
    return x_train, x_test, y_train, y_test


# # Reduction of component using PCA

# In[2]:

# import the iris dataset
iris = datasets.load_iris()

# Reduce the iris data set to 2 components
X_reduced = PCA(n_components=2).fit_transform(iris.data)
y = iris.target

# Create train_test_result
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, 0.2)


# # Finding Accuracy, precision, recall, F1 for "Uniform" one (k= 1to 50)

# In[3]:

# Creating a list to store accuracy, micro & macro averaged precision, recall & f1 score for all 'K' values.
accuracy=[]

precision_micro=[]
recall_micro=[]
f1_score_micro=[]

precision_macro=[]
recall_macro=[]
f1_score_macro=[]

# Creating a loop to get all accuracy, micro & macro averaged precision, recall & f1 score from k=1 to 50. 
for K in range(50):
    
    # Increasing the value of 'k' because neighbour should startn from 1 instead of 0.
    K_value = K+1
    
    #Create classifier object
    knn = KNeighborsClassifier(n_neighbors = K_value, weights='uniform')

    #train the classifier(fit the estimator using the training data)
    knn.fit(X_train, y_train)

    #Predict the test data
    class_predict=knn.predict(X_test)

    # Inserting a code to store Accuracy for all neighboure from 1 to 50
    accuracy.append(metrics.accuracy_score(y_test, class_predict)*100)
    
    # Inserting a code to store Micro averaged precision, recall, f1 score for all neighboure from 1 to 50
    precision_micro.append(metrics.precision_score(y_test, class_predict, average='micro')*100)
    recall_micro.append(metrics.recall_score(y_test, class_predict, average='micro')*100)
    f1_score_micro.append(metrics.f1_score(y_test, class_predict, average='micro')*100)
    
    # Inserting a code to store Macro averaged precision, recall, f1 score for all neighboure from 1 to 50
    precision_macro.append(metrics.precision_score(y_test, class_predict, average='macro')*100)
    recall_macro.append(metrics.recall_score(y_test, class_predict, average='macro')*100)
    f1_score_macro.append(metrics.f1_score(y_test, class_predict, average='macro')*100)

# Rounding off the accuracy up to 2 decimal point 
Accuracy = [ '%.2f' % elem for elem in accuracy]
print('\n\nAccuracy of all the classes from k = 1 to 50 are :-\n',Accuracy)

# Rounding off the Micro averaged precision, recall, f1 score up to 2 decimal point
Precision_micro = [ '%.2f' % elem for elem in precision_micro]
print('\n\nMicro Averaged Precision of all the classes from k = 1 to 50 are :-\n',Precision_micro)
Recall_micro = [ '%.2f' % elem for elem in recall_micro]
print('\nMicro Averaged Recall of all the classes from k = 1 to 50 are :-\n',Recall_micro)
F1_score_micro = [ '%.2f' % elem for elem in f1_score_micro]
print('\nMicro Averaged F1 score of all the classes from k = 1 to 50 are :-\n\n',F1_score_micro)

# Rounding off the Macro averaged precision, recall, f1 score up to 2 decimal point
Precision_macro = [ '%.2f' % elem for elem in precision_macro]
print('\n\nMacro Averaged Precision of all the classes from k = 1 to 50 are :-\n',Precision_macro)
Recall_macro = [ '%.2f' % elem for elem in recall_macro]
print('\nMacro Averaged Recall of all the classes from k = 1 to 50 are :-\n',Recall_macro)
F1_score_macro = [ '%.2f' % elem for elem in f1_score_macro]
print('\nMacro Averaged F1 score of all the classes from k = 1 to 50 are :-\n\n',F1_score_macro)


# # Individual plots b/w accuracy, precision, recall, f1, and no. of neighbors for 2 components || Uniform model

# In[4]:

# Inserting a code to create X-axis, K_neighbours.
K_neighbours=list(range(1,51,1))

# Inserting a code to increase the size of plot.
plt.figure(figsize=(16,22))

# Inserting a code to create a list of y-values.
y=[Accuracy, Precision_macro, Recall_macro, F1_score_macro, Precision_micro, Recall_micro, F1_score_micro]

# Inserting code to create a list of y-label.
y1=["Accuracy", "Precision (macro)", "Recall (macro)", "F1 score (macro)", 
    "Precision (micro)", "Recall (micro)", "F1 score (micro)"]

# Inserting code to create a for loop to plot all the graphs.
for i in range(0,len(y)):
    
    # Inserting code to set the subplot.
    plt.subplot(4,2,i+1)
    
    # Inserting code to plot the graph.
    plt.plot(K_neighbours, y[i], ls='-', color='red')
    
    # Inserting code to label the x-axis & y-axis.
    plt.ylabel(y1[i], fontsize=11)
    plt.xlabel('Number of neighbors(K)\n', fontsize=11)
    
    # Inserting code to write the title in the plots.
    plt.title("Number of neighbors(k) Vs "+y1[i], y=0.99, fontweight='bold', fontsize=12)


plt.show()


# NOTE:-
# As, we can see in the below plots, accuracy, micro averaged precision, recall & f1 are same. 
# However, if we see on accuracy & macro averaged precision, recall, f1, there is slight difference 
# at many points.


# # Plot b/w accuracy, macro averaged precision, recall, F1 vs no. of neighbours || Uniform classifier
# 

# In[5]:

# Inserting a code to increase the size of plot.
plt.figure(figsize=(14,10))


# Inserting a code to make a plot.
plt.plot(K_neighbours, Accuracy, 'bo', ls='-',  label='Accuracy')
plt.plot(K_neighbours, Precision_macro,'ro', label='Precision macro')
plt.plot(K_neighbours, Recall_macro, 'mo', label='Recall macro')
plt.plot(K_neighbours, F1_score_macro, 'go', label='F1 score macro')


# Inserting a code to label the x-axis & y-axis
plt.ylabel('Accuracy, Precision, Recall, & F1-score (Macro)', fontsize=14,fontweight='bold')
plt.xlabel('Number of neighbors(K)\n', fontsize=14,fontweight='bold')


# Inserting a code to write the title
plt.title(" \n         Plot b/w Accuracy, Precision, Recall, & F1-score Vs 'Number of neighbors'\n "+
          "for Uniform model (Macro averaged)", fontsize=20,fontweight='bold')


plt.legend(loc='best')
plt.show()


# Note:-
# if for any particular value of 'k', precision or recall is not visible in the beolow graph,
# that means precision & recall is similar to f1 score & showing as a green spot.


# # Plot b/w accuracy, micro averaged precision, recall, F1 vs no. of neighbours || Uniform classifier

# In[6]:

# Inserting a code to increase the size of plot.
plt.figure(figsize=(14,10))

# Inserting a code to make a plot.
plt.plot(K_neighbours, Accuracy, color="red", label='Accuracy')
plt.plot(K_neighbours, Precision_micro, color="red", label='Precision micro')
plt.plot(K_neighbours, Recall_micro, color="red",  label='Recall micro')
plt.plot(K_neighbours, F1_score_micro, color="red", label='F1 score micro')# linestyle = 'solid'

# Inserting a code to make a plot.
plt.ylabel('Accuracy, Precision, Recall, & F1-score (Micro)', fontsize=14,fontweight='bold')
plt.xlabel('Number of neighbors(K)\n', fontsize=14,fontweight='bold')

# Inserting a code to write the title
plt.title(" \n         Plot b/w Accuracy, Precision, Recall, & F1-score Vs 'Number of neighbors'\n "+
          "for Uniform model (Micro averaged)", fontsize=20,fontweight='bold')

plt.legend(loc='best')
plt.show()


# NOTE:-
# We can clearly see in the below graph, precision, recall & f1 are same in case of micro averaged.


# # Best & Worst classifier in Uniform Model

# In[7]:

# Inserting code to create a list to store neighbor values at best & worst accuracy
Best_uniform, Worst_uniform = [],[] 

#Inserting a code to get all the neighbour values for the best & worst accuracy
for i in range(0,len(Accuracy)):
    if Accuracy[i]==max(Accuracy):
        Best_uniform.append(i+1)
    if Accuracy[i]==min(Accuracy):
        Worst_uniform.append(i+1)
        
Best_uniform_value=max(Accuracy)
Worst_uniform_value=min(Accuracy)
        
print("\nBest classifier for uniform model is showing "+ Best_uniform_value +" at K="+str(Best_uniform))
print("\nWorst classifier for uniform model is showing "+ Worst_uniform_value +" at K="+str(Worst_uniform)+"\n")


# Note:-
# Here, n below classification region, we are showing only one value of k for best classifier & one value of 
# k for Worst classifier 


# # Classification region for the best & worst classifier in the above "Uniform" case

# In[8]:

# As we can see in above plot Best classifier is for k_vlue=[2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18,20,32,34,36]
# and worst classifier is for K_vlue=[1, 19, 21, 39, 40, 41] based on high & low accuracy

# We can take any value right now to show the best & worst classifier


# In[12]:


#Inserting a code to convert tuple i.e. X_test into the format of numpy.ndarray i.e. X_reduced 
#to plot the classification region and testing data
a,b=[],[]
for i in range(0,len(X_test)):
    a.append(X_test[i][0])
    b.append(X_test[i][1])
A,B=np.array(a),np.array(b)
xtest=np.column_stack((A,B))

# Inserting code to take any value of k for best accuracy & worst accuracy.
k_value=[random.choice(Best_uniform),random.choice(Worst_uniform)]

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

h = .02  # step size in the mesh

# Creating a loop to get the classification region for best & worst classifier.
for k in k_value:
    
    # Inserting code to mention the type of classifier.
    if k==k_value[0]:
        classif_="Best Classifier"
    elif k==k_value[1]:
        classif_="Worst Classifier"

    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors=k, weights='uniform')
    
    # Fitting the training data.
    clf.fit(X_train, y_train)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = xtest[:, 0].min() - 1, xtest[:, 0].max() + 1
    y_min, y_max = xtest[:, 1].min() - 1, xtest[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(xtest[:, 0], xtest[:, 1], c=y_test, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification for "+classif_+" at k="+str(k))

    plt.show()


# # Finding Accuracy, precision, recall, F1 for component = 2 ("Distance", k= 1 to 50)

# In[13]:

# Reduce the iris data set to 2 components
X_reduced = PCA(n_components=2).fit_transform(iris.data)
y = iris.target

# Create train_test_result
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, 0.2)


# In[14]:

# Creating a list to store accuracy, micro & macro averaged precision, recall & f1 score for all 'K' values.
accuracy=[]

precision_micro=[]
recall_micro=[]
f1_score_micro=[]

precision_macro=[]
recall_macro=[]
f1_score_macro=[]

# Creating a loop to get all accuracy, micro & macro averaged precision, recall & f1 score from k=1 to 50.
for K in range(50):
    
    # Increasing the value of 'k' because neighbour should start from 1 instead of 0.
    K_value = K+1    

    #Create classifier object
    knn = KNeighborsClassifier(n_neighbors = K_value, weights='distance')

    #train the classifier(fit the estimator using the training data)
    knn.fit(X_train, y_train)

    #Predict the test data
    class_predict=knn.predict(X_test)

    
    # Inserting a code to store Accuracy for all neighboure from 1 to 50
    accuracy.append(metrics.accuracy_score(y_test, class_predict)*100)
    
    # Inserting a code to store Micro averaged precision, recall, f1 score for all neighboure from 1 to 50
    precision_micro.append(metrics.precision_score(y_test, class_predict, average='micro')*100)
    recall_micro.append(metrics.recall_score(y_test, class_predict, average='micro')*100)
    f1_score_micro.append(metrics.f1_score(y_test, class_predict, average='micro')*100)

    # Inserting a code to store Macro averaged precision, recall, f1 score for all neighboure from 1 to 50
    precision_macro.append(metrics.precision_score(y_test, class_predict, average='macro')*100)
    recall_macro.append(metrics.recall_score(y_test, class_predict, average='macro')*100)
    f1_score_macro.append(metrics.f1_score(y_test, class_predict, average='macro')*100)

# Rounding off the accuracy up to 2 decimal point 
Accuracy = [ '%.2f' % elem for elem in accuracy]
print('\n\nAccuracy of all the classes from k = 1 to 50 are :-\n\n',Accuracy)

# Rounding off the Micro averaged precision, recall, f1 score up to 2 decimal point
Precision_micro = [ '%.2f' % elem for elem in precision_micro]
print('\n\nMicro Averaged Precision of all the classes from k = 1 to 50 are :-\n\n',Precision_micro)
Recall_micro = [ '%.2f' % elem for elem in recall_micro]
print('\n\nMicro Averaged Recall of all the classes from k = 1 to 50 are :-\n\n',Recall_micro)
F1_score_micro = [ '%.2f' % elem for elem in f1_score_micro]
print('\n\nMicro Averaged F1 score of all the classes from k = 1 to 50 are :-\n\n',F1_score_micro)

# Rounding off the Macro averaged precision, recall, f1 score up to 2 decimal point
Precision_macro = [ '%.2f' % elem for elem in precision_macro]
print('\n\nMacro Averaged Precision of all the classes from k = 1 to 50 are :-\n\n',Precision_macro)
Recall_macro = [ '%.2f' % elem for elem in recall_macro]
print('\n\nMacro Averaged Recall of all the classes from k = 1 to 50 are :-\n\n',Recall_macro)
F1_score_macro = [ '%.2f' % elem for elem in f1_score_macro]
print('\n\nMacro Averaged F1 score of all the classes from k = 1 to 50 are :-\n\n',F1_score_macro)


# # Individual plots b/w accuracy, precision, recall, f1, and no. of neighbors || Distance model

# In[15]:

# Inserting a code to increase the size of plot.
plt.figure(figsize=(16,22))

# Inserting a code to create a list of y-values.
y=[Accuracy, Precision_macro, Recall_macro, F1_score_macro, Precision_micro, Recall_micro, F1_score_micro]

# Inserting code to create a list of y-label.
y1=["Accuracy", "Precision (macro)", "Recall (macro)", "F1 score (macro)", 
    "Precision (micro)", "Recall (micro)", "F1 score (micro)"]

# Inserting code to create a for loop to plot all the graphs.
for i in range(0,len(y)):
    
    # Inserting code to set the subplot.
    plt.subplot(4,2,i+1)
    
    # Inserting code to plot the graph.
    plt.plot(K_neighbours, y[i], ls='-', color='red')
    
    # Inserting code to label the x-axis & y-axis.
    plt.ylabel(y1[i], fontsize=11)
    plt.xlabel('Number of neighbors(K)\n', fontsize=11)
    
    # Inserting code to write the title in the plots.
    plt.title("Number of neighbors(k) Vs "+y1[i], y=0.99, fontweight='bold', fontsize=12)
    
plt.show()



# NOTE:-
# As, we can see in the below plots, accuracy, micro averaged precision, recall & f1 are same. 
# However, if we see on accuracy & macro averaged precision, recall, f1, there is slight difference 
# at many points.


# # Plot b/w accuracy, macro averaged precision, recall, F1 vs no. of neighbours || Distance classifier

# In[16]:

# Inserting a code to create X-axis, K_neighbours.
K_neighbours=list(range(1,51,1))


# Inserting a code to increase the size of plot.
plt.figure(figsize=(14,10))


# Inserting a code to make a plot.
plt.plot(K_neighbours, Accuracy, 'bo',  ls='-', label='Accuracy')
plt.plot(K_neighbours, Precision_macro,'ro', label='Precision macro')
plt.plot(K_neighbours, Recall_macro, 'y^', label='Recall macro')
plt.plot(K_neighbours, F1_score_macro, 'go', label='F1 score macro')


# Inserting a code to label the x-axis & y-axis
plt.ylabel('Accuracy, Precision, Recall, & F1-score (Macro)', fontsize=14,fontweight='bold')
plt.xlabel('Number of neighbors(K)\n', fontsize=14,fontweight='bold')


# Inserting a code to write the title
plt.title(" \n         Plot b/w Accuracy, Precision, Recall, & F1-score Vs 'Number of neighbors'\n "+
          "for Distance model (Macro averaged)", fontsize=20,fontweight='bold')

plt.legend(loc='best')
plt.show()


# Note:-
# if for any particular value of 'k', precision or recall is not visible in the beolow graph,
# that means precision & recall is similar to f1 score & showing as a green spot.


# # Plot b/w accuracy, micro averaged precision, recall, F1 vs no. of neighbours || Distance classifier

# In[17]:

# Inserting a code to increase the size of plot.
plt.figure(figsize=(14,10))

# Inserting a code to make a plot.
plt.plot(K_neighbours, Accuracy, color="red", label='Accuracy')
plt.plot(K_neighbours, Precision_micro, color="red", label='Precision micro')
plt.plot(K_neighbours, Recall_micro, color="red",  label='Recall micro')
plt.plot(K_neighbours, F1_score_micro, color="red", label='F1 score micro')# linestyle = 'solid'

# Inserting a code to make a plot.
plt.ylabel('Accuracy, Precision, Recall, & F1-score (Micro)', fontsize=14,fontweight='bold')
plt.xlabel('Number of neighbors(K)\n', fontsize=14,fontweight='bold')

# Inserting a code to write the title
plt.title(" \n         Plot b/w Accuracy, Precision, Recall, & F1-score Vs 'Number of neighbors'\n "+
          "for Distance model (Micro averaged)", fontsize=20,fontweight='bold')

plt.legend(loc='best')
plt.show()


# NOTE:-
# We can clearly see in the below graph, precision, recall & f1 are same in case of micro averaged.


# # Best & Worst classifier for Distance Model

# In[18]:

# Inserting code to create a list to store neighbor values at best & worst accuracy
Best_distance, Worst_distance = [],[] 

#Inserting a code to get all the neighbour values for the best & worst accuracy
for i in range(0,len(Accuracy)):
    if Accuracy[i]==max(Accuracy):
        Best_distance.append(i+1)
    if Accuracy[i]==min(Accuracy):
        Worst_distance.append(i+1)
        
Best_distance_value=max(Accuracy)
Worst_distance_value=min(Accuracy)
        
print("\nBest classifier for Distance model is showing "+ Best_distance_value +" at K="+str(Best_distance))
print("\nWorst classifier for Distance model is showing "+ Worst_distance_value +" at K="+str(Worst_distance)+"\n")


# # Classification region for the best & worst classifier  || Distance one

# In[ ]:

# As we can see in above plot Best classifier is for k_vlue=10
# and worst classifier is for k_vlue=48 based on high & low accuracy


# In[20]:



#Inserting a code to convert tuple i.e. X_test into the format of numpy.ndarray i.e. X_reduced 
#to plot the classification region and testing data
a,b=[],[]
for i in range(0,len(X_test)):
    a.append(X_test[i][0])
    b.append(X_test[i][1])
A,B=np.array(a),np.array(b)
xtest=np.column_stack((A,B))

# Inserting code to take any value of k for best accuracy & worst accuracy.
k_value=[random.choice(Best_distance),random.choice(Worst_distance)]

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


h = .02  # step size in the mesh

# Creating a loop to get the classification region for best & worst classifier.
for k in k_value:
    
    # Inserting code to mention the type of classifier.
    if k==k_value[0]:
        classif_="Best Classifier"
    elif k==k_value[1]:
        classif_="Worst Classifier"

    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors=k, weights='distance')
    
    # Fitting the training data.
    clf.fit(X_train, y_train)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = xtest[:, 0].min() - 1, xtest[:, 0].max() + 1
    y_min, y_max = xtest[:, 1].min() - 1, xtest[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(xtest[:, 0], xtest[:, 1], c=y_test, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification for "+classif_+" at k="+str(k))

    plt.show()


# # Question 2 :- Type 2
# #                        Use PCA to reduce the iris data set to 2 components.
# #                        Plot accuracy, precision, recall, and f1-score ("For each class individually") vs. the number of neighbors with a 20% test split on the iris data 
# #                        And, plot the classification region for best & worst classifier

# 
# 
# 
# # Uniform model:-

# In[25]:


# Reduce the iris data set to 2 components
X_reduced = PCA(n_components=2).fit_transform(iris.data)
y = iris.target

# Create train_test_result
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, 0.2)


# Creating a list to store accuracy, micro & macro averaged precision, recall & f1 score for all 'K' values.
accuracy=[]

precision_class1=[]
precision_class2=[]
precision_class3=[]

recall_class1=[]
recall_class2=[]
recall_class3=[]

f1_score_class1=[]
f1_score_class2=[]
f1_score_class3=[]

# Creating a loop to get all accuracy, micro & macro averaged precision, recall & f1 score from k=1 to 50.
for K in range(50):
    
    # Increasing the value of 'k' because neighbour should start from 1 instead of 0.
    K_value = K+1
    
    #Create classifier object
    knn = KNeighborsClassifier(n_neighbors = K_value, weights='uniform')

    #train the classifier(fit the estimator using the training data)
    knn.fit(X_train, y_train)

    #Predict the test data
    class_predict=knn.predict(X_test)


    #Create confusion matrix to calculate the accuracy, precision, recall, f1 score for each class
    Matrix=metrics.confusion_matrix(y_test, class_predict)
        
    
    #accuracy
    accuracy.append((metrics.accuracy_score(y_test, class_predict))*100)
    
    #Precision for class 1 (i.e 0 or 'setosa')
    precision_class1.append((Matrix[0][0]/(Matrix[0][0]+Matrix[1][0]+Matrix[2][0]))*100)

    #Precision for class 2 (i.e 1 or 'versicolor')
    precision_class2.append((Matrix[1][1]/(Matrix[0][1]+Matrix[1][1]+Matrix[2][1]))*100)
    
    #Precision for class 3 (i.e 2 or 'virginica')
    precision_class3.append((Matrix[2][2]/(Matrix[0][2]+Matrix[1][2]+Matrix[2][2]))*100)

    
    #Recall for class 1 (i.e 0 or 'setosa')
    recall_class1.append((Matrix[0][0]/(Matrix[0][0]+Matrix[0][1]+Matrix[0][2]))*100)

    #Recall for class 2 (i.e 1 or 'versicolor')
    recall_class2.append((Matrix[1][1]/(Matrix[1][0]+Matrix[1][1]+Matrix[1][2]))*100)
    
    #Recall for class 3 (i.e 2 or 'virginica')
    recall_class3.append((Matrix[2][2]/(Matrix[2][0]+Matrix[2][1]+Matrix[2][2]))*100)
 

    #F1-score for class 1 (i.e 0 or 'setosa')
    f1_score_class1.append((2*precision_class1[K]*recall_class1[K])/(precision_class1[K]+recall_class1[K]))
    
    #Recall for class 2 (i.e 1 or 'versicolor')
    f1_score_class2.append((2*precision_class2[K]*recall_class2[K])/(precision_class2[K]+recall_class2[K]))
    
    #Recall for class 3 (i.e 2 or 'virginica')
    f1_score_class3.append((2*precision_class3[K]*recall_class3[K])/(precision_class3[K]+recall_class3[K]))

# Rounding off the accuracy up to 2 decimal point 
Accuracy = [ '%.2f' % elem for elem in accuracy]
print('\n\nAccuracy(k = 1 to 50) are :-\n\n',Accuracy)

# Rounding off the precision up to 2 decimal point for class 1, class 2 & class 3
Precision_class1 = [ '%.2f' % elem for elem in precision_class1]
print('\n\nPrecision for Class1(k = 1 to 50) :-\n',Precision_class1)

Precision_class2 = [ '%.2f' % elem for elem in precision_class2]
print('\nPrecision for Class2(k = 1 to 50) :-\n',Precision_class2)

Precision_class3 = [ '%.2f' % elem for elem in precision_class3]
print('\nPrecision for Class3(k = 1 to 50) :-\n\n',Precision_class3)

# Rounding off the recall up to 2 decimal point for class 1, class 2 & class 3
Recall_class1 = [ '%.2f' % elem for elem in recall_class1]
print('\n\nRecall for Class1(k = 1 to 50) :-\n',Recall_class1)

Recall_class2 = [ '%.2f' % elem for elem in recall_class2]
print('\nRecall for Class2(k = 1 to 50) :-\n',Recall_class2)

Recall_class3 = [ '%.2f' % elem for elem in recall_class3]
print('\nRecall for Class3(k = 1 to 50) :-\n\n',Recall_class3)

# Rounding off the f1 score up to 2 decimal point for class 1, class 2 & class 3
F1_score_class1 = [ '%.2f' % elem for elem in f1_score_class1]
print('\n\nF1 score for class1(k = 1 to 50) :-\n',F1_score_class1)

F1_score_class2 = [ '%.2f' % elem for elem in f1_score_class2]
print('\nF1 score for class2(k = 1 to 50) :-\n',F1_score_class2)

F1_score_class3 = [ '%.2f' % elem for elem in f1_score_class3]
print('\nF1 score for class3(k = 1 to 50) :-\n',F1_score_class3)


# # Plot for class 1 for reduced component || Uniform model

# In[26]:

# Inserting a code to increase the size of plot.
plt.figure(figsize=(16,22))

# Inserting a code to create a list of y-values.
y=[Accuracy, Precision_class1, Recall_class1, F1_score_class1]

# Inserting code to create a list of y-label.
y1=["Accuracy", "Precision (class 1)", "Recall (class 1)", "F1 score (class 1)"]

# Inserting code to create a for loop to plot all the graphs.
for i in range(0,len(y)):
    
    # Inserting code to set the subplot.
    plt.subplot(4,2,i+1)
    
    # Inserting code to plot the graph.
    plt.plot(K_neighbours, y[i], ls='-', color='red')
    
    # Inserting code to label the x-axis & y-axis.
    plt.ylabel(y1[i], fontsize=11)
    plt.xlabel('Number of neighbors(K)\n', fontsize=11)
    
    # Inserting code to write the title in the plots.
    plt.title("Number of neighbors(k) Vs "+y1[i], y=0.99, fontweight='bold', fontsize=12)
    
plt.show()


# In[27]:

# Inserting a code to increase the size of plot.
plt.figure(figsize=(14,10))


# Inserting a code to make a plot.
plt.plot(K_neighbours, Accuracy, 'bo', ls='-',label='Accuracy')
plt.plot(K_neighbours, Precision_class1, 'ro', label='Precision class 1')
plt.plot(K_neighbours, Recall_class1,'mo',  label='Recall class 1')
plt.plot(K_neighbours, F1_score_class1,'go', label='F1 score class 1')# linestyle = 'solid'


# Inserting a code to label the x-axis & y-axis
plt.ylabel('Accuracy, Precision, Recall, & F1-score (Class 1)', fontsize=14,fontweight='bold')
plt.xlabel('Number of neighbors(K)\n', fontsize=14,fontweight='bold')


# Inserting a code to write the title
plt.title(" \n         Plot b/w Accuracy, Precision, Recall, & F1-score Vs 'Number of neighbors'\n "+
          "for Uniform model (Class 1)", fontsize=20,fontweight='bold')

plt.legend(loc='best')
plt.show()




# Note:-
# if for any particular value of 'k', precision or recall is not visible in the beolow graph,
# that means precision & recall is similar to f1 score & showing as a green spot.


# # Plot for Class 2 for reduced component || Uniform model

# In[28]:

# Inserting a code to increase the size of plot.
plt.figure(figsize=(16,22))

# Inserting a code to create a list of y-values.
y=[Accuracy, Precision_class2, Recall_class2, F1_score_class2]

# Inserting code to create a list of y-label.
y1=["Accuracy", "Precision (class 2)", "Recall (class 2)", "F1 score (class 2)"]

# Inserting code to create a for loop to plot all the graphs.
for i in range(0,len(y)):
    
    # Inserting code to set the subplot.
    plt.subplot(4,2,i+1)
    
    # Inserting code to plot the graph.
    plt.plot(K_neighbours, y[i], ls='-', color='red')
    
    # Inserting code to label the x-axis & y-axis.
    plt.ylabel(y1[i], fontsize=11)
    plt.xlabel('Number of neighbors(K)\n', fontsize=11)
    
    # Inserting code to write the title in the plots.
    plt.title("Number of neighbors(k) Vs "+y1[i], y=0.99, fontweight='bold', fontsize=12)
    
plt.show()


# In[29]:

# Inserting a code to increase the size of plot.
plt.figure(figsize=(14,10))


# Inserting a code to make a plot.
plt.plot(K_neighbours, Accuracy, 'bo',ls='-', label='Accuracy')
plt.plot(K_neighbours, Precision_class2, 'ro', label='Precision class 2')
plt.plot(K_neighbours, Recall_class2,'mo',  label='Recall class 2')
plt.plot(K_neighbours, F1_score_class2,'go', label='F1 score class 2')# linestyle = 'solid'


# Inserting a code to label the x-axis & y-axis
plt.ylabel('Accuracy, Precision, Recall, & F1-score (Class 2)', fontsize=14,fontweight='bold')
plt.xlabel('Number of neighbors(K)\n', fontsize=14,fontweight='bold')


# Inserting a code to write the title
plt.title(" \n         Plot b/w Accuracy, Precision, Recall, & F1-score Vs 'Number of neighbors'\n "+
          "for Uniform model (Class 2)", fontsize=20,fontweight='bold')

plt.legend(loc='best')
plt.show()




# Note:-
# if for any particular value of 'k', precision or recall is not visible in the beolow graph,
# that means precision & recall is similar to f1 score & showing as a green spot.


# # Plot for Class 3 for reduced component || Uniform model

# In[30]:

# Inserting a code to increase the size of plot.
plt.figure(figsize=(16,22))

# Inserting a code to create a list of y-values.
y=[Accuracy, Precision_class3, Recall_class3, F1_score_class3]

# Inserting code to create a list of y-label.
y1=["Accuracy", "Precision (class 3)", "Recall (class 3)", "F1 score (class 3)"]

# Inserting code to create a for loop to plot all the graphs.
for i in range(0,len(y)):
    
    # Inserting code to set the subplot.
    plt.subplot(4,2,i+1)
    
    # Inserting code to plot the graph.
    plt.plot(K_neighbours, y[i], ls='-', color='red')
    
    # Inserting code to label the x-axis & y-axis.
    plt.ylabel(y1[i], fontsize=11)
    plt.xlabel('Number of neighbors(K)\n', fontsize=11)
    
    # Inserting code to write the title in the plots.
    plt.title("Number of neighbors(k) Vs "+y1[i], y=0.99, fontweight='bold', fontsize=12)
    
plt.show()


# In[31]:

# Inserting a code to increase the size of plot.
plt.figure(figsize=(14,10))


# Inserting a code to make a plot.
plt.plot(K_neighbours, Accuracy, 'bo', label='Accuracy')
plt.plot(K_neighbours, Precision_class3, 'ro', label='Precision class 3')
plt.plot(K_neighbours, Recall_class3,'mo',  label='Recall class 3')
plt.plot(K_neighbours, F1_score_class3,'go', label='F1 score class 3')


# Inserting a code to label the x-axis & y-axis
plt.ylabel('Accuracy, Precision, Recall, & F1-score (Class 3)', fontsize=14,fontweight='bold')
plt.xlabel('Number of neighbors(K)\n', fontsize=14,fontweight='bold')


# Inserting a code to write the title
plt.title(" \n         Plot b/w Accuracy, Precision, Recall, & F1-score Vs 'Number of neighbors'\n "+
          "for Uniform model (Class 3)", fontsize=20,fontweight='bold')

plt.legend(loc='best')
plt.show()




# Note:-
# if for any particular value of 'k', precision or recall is not visible in the beolow graph,
# that means precision & recall is similar to f1 score & showing as a green spot.


# # Finding Accuracy, precision, recall, F1 for reduced component ("Distance", k= 1to 50) for each classes i.e. Class 1, Class 2, Class 3

# In[35]:

# Reduce the iris data set to 2 components
X_reduced = PCA(n_components=2).fit_transform(iris.data)
y = iris.target

# Create train_test_result
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, 0.2)


# Creating a list to store accuracy, micro & macro averaged precision, recall & f1 score for all 'K' values.
accuracy=[]

precision_class1=[]
precision_class2=[]
precision_class3=[]

recall_class1=[]
recall_class2=[]
recall_class3=[]

f1_score_class1=[]
f1_score_class2=[]
f1_score_class3=[]


# Creating a loop to get all accuracy, micro & macro averaged precision, recall & f1 score from k=1 to 50.
for K in range(50):
    
    # Increasing the value of 'k' because neighbour should start from 1 instead of 0.
    K_value = K+1  
    
    #Create classifier object
    knn = KNeighborsClassifier(n_neighbors = K_value, weights='distance')

    #train the classifier(fit the estimator using the training data)
    knn.fit(X_train, y_train)

    #Predict the test data
    class_predict=knn.predict(X_test)


    #Create confusion matrix to calculate the accuracy, precision, recall, f1 score for each class
    Matrix=metrics.confusion_matrix(y_test, class_predict)
    
    
    
    #accuracy
    accuracy.append((metrics.accuracy_score(y_test, class_predict))*100)
    
    #Precision for class 1 (i.e 0 or 'setosa')
    precision_class1.append((Matrix[0][0]/(Matrix[0][0]+Matrix[1][0]+Matrix[2][0]))*100)

    #Precision for class 2 (i.e 1 or 'versicolor')
    precision_class2.append((Matrix[1][1]/(Matrix[0][1]+Matrix[1][1]+Matrix[2][1]))*100)
    
    #Precision for class 3 (i.e 2 or 'virginica')
    precision_class3.append((Matrix[2][2]/(Matrix[0][2]+Matrix[1][2]+Matrix[2][2]))*100)

    
    #Recall for class 1 (i.e 0 or 'setosa')
    recall_class1.append((Matrix[0][0]/(Matrix[0][0]+Matrix[0][1]+Matrix[0][2]))*100)

    #Recall for class 2 (i.e 1 or 'versicolor')
    recall_class2.append((Matrix[1][1]/(Matrix[1][0]+Matrix[1][1]+Matrix[1][2]))*100)
    
    #Recall for class 3 (i.e 2 or 'virginica')
    recall_class3.append((Matrix[2][2]/(Matrix[2][0]+Matrix[2][1]+Matrix[2][2]))*100)
 

    #F1-score for class 1 (i.e 0 or 'setosa')
    f1_score_class1.append((2*precision_class1[K]*recall_class1[K])/(precision_class1[K]+recall_class1[K]))
    
    #Recall for class 2 (i.e 1 or 'versicolor')
    f1_score_class2.append((2*precision_class2[K]*recall_class2[K])/(precision_class2[K]+recall_class2[K]))
    
    #Recall for class 3 (i.e 2 or 'virginica')
    f1_score_class3.append((2*precision_class3[K]*recall_class3[K])/(precision_class3[K]+recall_class3[K]))

# Rounding off the accuracy up to 2 decimal point.
Accuracy = [ '%.2f' % elem for elem in accuracy]
print('\n\nAccuracy(k = 1 to 50) are :-\n\n',Accuracy)

# Rounding off the precision up to 2 decimal point for class 1, class 2 & class 3
Precision_class1 = [ '%.2f' % elem for elem in precision_class1]
print('\n\nPrecision for Class1(k = 1 to 50) :-\n',Precision_class1)

Precision_class2 = [ '%.2f' % elem for elem in precision_class2]
print('\nPrecision for Class2(k = 1 to 50) :-\n',Precision_class2)

Precision_class3 = [ '%.2f' % elem for elem in precision_class3]
print('\nPrecision for Class3(k = 1 to 50) :-\n\n',Precision_class3)

# Rounding off the recall up to 2 decimal point for class 1, class 2 & class 3
Recall_class1 = [ '%.2f' % elem for elem in recall_class1]
print('\n\nRecall for Class1(k = 1 to 50) :-\n',Recall_class1)

Recall_class2 = [ '%.2f' % elem for elem in recall_class2]
print('\nRecall for Class2(k = 1 to 50) :-\n',Recall_class2)

Recall_class3 = [ '%.2f' % elem for elem in recall_class3]
print('\nRecall for Class3(k = 1 to 50) :-\n\n',Recall_class3)

# Rounding off the f1 score up to 2 decimal point for class 1, class 2 & class 3
F1_score_class1 = [ '%.2f' % elem for elem in f1_score_class1]
print('\n\nF1 score for class1(k = 1 to 50) :-\n',F1_score_class1)

F1_score_class2 = [ '%.2f' % elem for elem in f1_score_class2]
print('\nF1 score for class2(k = 1 to 50) :-\n',F1_score_class2)

F1_score_class3 = [ '%.2f' % elem for elem in f1_score_class3]
print('\nF1 score for class3(k = 1 to 50) :-\n',F1_score_class3)


# # Plot for 'class 1' for reduced component=2 || Distance model

# In[36]:

# Inserting a code to increase the size of plot.
plt.figure(figsize=(16,22))

# Inserting a code to create a list of y-values.
y=[Accuracy, Precision_class1, Recall_class1, F1_score_class1]

# Inserting code to create a list of y-label.
y1=["Accuracy", "Precision (class 1)", "Recall (class 1)", "F1 score (class 1)"]

# Inserting code to create a for loop to plot all the graphs.
for i in range(0,len(y)):
    
    # Inserting code to set the subplot.
    plt.subplot(4,2,i+1)
    
    # Inserting code to plot the graph.
    plt.plot(K_neighbours, y[i], ls='-', color='red')
    
    # Inserting code to label the x-axis & y-axis.
    plt.ylabel(y1[i], fontsize=11)
    plt.xlabel('Number of neighbors(K)\n', fontsize=11)
    
    # Inserting code to write the title in the plots.
    plt.title("Number of neighbors(k) Vs "+y1[i], y=0.99, fontweight='bold', fontsize=12)
    
plt.show()



# In[37]:

# Inserting a code to increase the size of plot.
plt.figure(figsize=(14,10))


# Inserting a code to make a plot.
plt.plot(K_neighbours, Accuracy, 'bo', ls='-',label='Accuracy')
plt.plot(K_neighbours, Precision_class1, 'ro', label='Precision class 1')
plt.plot(K_neighbours, Recall_class1,'mo',  label='Recall class 1')
plt.plot(K_neighbours, F1_score_class1,'go', label='F1 score class 1')# linestyle = 'solid'


# Inserting a code to label the x-axis & y-axis
plt.ylabel('Accuracy, Precision, Recall, & F1-score (Class 1)', fontsize=14,fontweight='bold')
plt.xlabel('Number of neighbors(K)\n', fontsize=14,fontweight='bold')


# Inserting a code to write the title
plt.title(" \n         Plot b/w Accuracy, Precision, Recall, & F1-score Vs 'Number of neighbors'\n "+
          "for Distance model (Class 1)", fontsize=20,fontweight='bold')

plt.legend(loc='best')
plt.show()


# Note:-
# if for any particular value of 'k', precision or recall is not visible in the beolow graph,
# that means precision & recall is similar to f1 score & showing as a green spot.


# # Plot for 'class 2' for reduced component=2 || Distance model

# In[38]:

# Inserting a code to increase the size of plot.
plt.figure(figsize=(16,22))

# Inserting a code to create a list of y-values.
y=[Accuracy, Precision_class2, Recall_class2, F1_score_class2]

# Inserting code to create a list of y-label.
y1=["Accuracy", "Precision (class 2)", "Recall (class 2)", "F1 score (class 2)"]

# Inserting code to create a for loop to plot all the graphs.
for i in range(0,len(y)):
    
    # Inserting code to set the subplot.
    plt.subplot(4,2,i+1)
    
    # Inserting code to plot the graph.
    plt.plot(K_neighbours, y[i], ls='-', color='red')
    
    # Inserting code to label the x-axis & y-axis.
    plt.ylabel(y1[i], fontsize=11)
    plt.xlabel('Number of neighbors(K)\n', fontsize=11)
    
    # Inserting code to write the title in the plots.
    plt.title("Number of neighbors(k) Vs "+y1[i], y=0.99, fontweight='bold', fontsize=12)
    
plt.show()


# In[39]:

# Inserting a code to increase the size of plot.
plt.figure(figsize=(14,10))


# Inserting a code to make a plot.
plt.plot(K_neighbours, Accuracy, 'bo', ls='-',label='Accuracy')
plt.plot(K_neighbours, Precision_class2, 'ro', label='Precision class 2')
plt.plot(K_neighbours, Recall_class2,'mo',  label='Recall class 2')
plt.plot(K_neighbours, F1_score_class2,'go', label='F1 score class 2')# linestyle = 'solid'


# Inserting a code to label the x-axis & y-axis
plt.ylabel('Accuracy, Precision, Recall, & F1-score (Class 2)', fontsize=14,fontweight='bold')
plt.xlabel('Number of neighbors(K)\n', fontsize=14,fontweight='bold')


# Inserting a code to write the title
plt.title(" \n         Plot b/w Accuracy, Precision, Recall, & F1-score Vs 'Number of neighbors'\n "+
          "for Distance model (Class 2)", fontsize=20,fontweight='bold')

plt.legend(loc='best')
plt.show()


# Note:-
# if for any particular value of 'k', precision or recall is not visible in the beolow graph,
# that means precision & recall is similar to f1 score & showing as a green spot.


# # Plot for 'class 3' for reduced component=2 || Distance model

# In[40]:

# Inserting a code to increase the size of plot.
plt.figure(figsize=(16,22))

# Inserting a code to create a list of y-values.
y=[Accuracy, Precision_class3, Recall_class3, F1_score_class3]

# Inserting code to create a list of y-label.
y1=["Accuracy", "Precision (class 3)", "Recall (class 3)", "F1 score (class 3)"]

# Inserting code to create a for loop to plot all the graphs.
for i in range(0,len(y)):
    
    # Inserting code to set the subplot.
    plt.subplot(4,2,i+1)
    
    # Inserting code to plot the graph.
    plt.plot(K_neighbours, y[i], ls='-', color='red')
    
    # Inserting code to label the x-axis & y-axis.
    plt.ylabel(y1[i], fontsize=11)
    plt.xlabel('Number of neighbors(K)\n', fontsize=11)
    
    # Inserting code to write the title in the plots.
    plt.title("Number of neighbors(k) Vs "+y1[i], y=0.99, fontweight='bold', fontsize=12)
    
plt.show()


# In[42]:

# Inserting a code to increase the size of plot.
plt.figure(figsize=(14,10))


# Inserting a code to make a plot.
plt.plot(K_neighbours, Accuracy, 'bo', ls='-',label='Accuracy')
plt.plot(K_neighbours, Precision_class3, 'ro', label='Precision class 3')
plt.plot(K_neighbours, Recall_class3,'mo',  label='Recall class 3')
plt.plot(K_neighbours, F1_score_class3,'go', label='F1 score class 3')# linestyle = 'solid'


# Inserting a code to label the x-axis & y-axis
plt.ylabel('Accuracy, Precision, Recall, & F1-score (Class 3)', fontsize=14,fontweight='bold')
plt.xlabel('Number of neighbors(K)\n', fontsize=14,fontweight='bold')


# Inserting a code to write the title
plt.title(" \n         Plot b/w Accuracy, Precision, Recall, & F1-score Vs 'Number of neighbors'\n "+
          "for Distance model (Class 3)", fontsize=20,fontweight='bold')

plt.legend(loc='best')
plt.show()


# Note:-
# if for any particular value of 'k', precision or recall is not visible in the beolow graph,
# that means precision & recall is similar to f1 score & showing as a green spot.


# # Questions & Final Conclusion
# 
# # Best & Worst Classifier

# In[43]:

# Inserting a code to get the best & worst classifier.
if max(Best_distance_value,Best_uniform_value)==Best_distance_value:
    k_neighbor1=Best_distance
    classif1="Distance Classifier"
if max(Best_distance_value,Best_uniform_value)==Best_uniform_value:
    k_neighbor1=Best_uniform
    classif1="Uniform Classifier"
if min(Worst_distance_value,Worst_uniform_value)==Worst_distance_value:
    k_neighbor2=Worst_distance
    classif2="Distance Classifier"
if min(Worst_distance_value,Worst_uniform_value)==Worst_uniform_value:
    k_neighbor2=Worst_uniform
    classif2="Uniform Classifier"
    
# Now, we can easily conclude that:-
print("\n\nAs we can clearly see above:-")
print("\nBest accuracy is coming "+ str(max(Best_distance_value,Best_uniform_value)) +" at n_neighbors('k') = "
      + str(k_neighbor1)+ " for "+classif1+ " and,")
print("\nWorst accuracy is coming "+ str(min(Worst_distance_value,Worst_uniform_value)) +" at n_neighbors('k') = "
      + str(k_neighbor2)+" for "+classif2)


NOTE:- We can choose any value of "K" that exists in the middle so that our model will not face any issue related to underfitting or Overfitting.


Therefore, we could say, Best model here is "Distance" one for K=1 and Worst model is "Uniform" one for K=41.


# # Bias & Variance

# In[ ]:

# What we are observing here is 
# Large "K" means simple model which is always consdidered as high bias and less variance. A higher K means more resilient to 
# outliers. The decision boundary would become smoother by increasing the value of K

# However, in case of smaller "k" value, will always get low bias but high variance.
# A small value for K provides the most flexible fit. Graphically, our decision boundary will be more jagged.

