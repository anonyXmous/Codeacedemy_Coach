
# coding: utf-8

# In[13]:

import sklearn as sk
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn.datasets as skd
import sklearn.model_selection

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split


# In[14]:

#load data from breast cancer dataset
breast_cancer_data = skd.load_breast_cancer()


# In[15]:

#Peek at the first row of data
breast_cancer_data.data[0]


# In[16]:

#Print the features names
print(breast_cancer_data.feature_names)


# In[17]:

#Was the very first data point tagged as malignant or benign?
print(breast_cancer_data.target)
print(breast_cancer_data.target_names)


# In[18]:

print("First data is benign (0)")


# In[19]:


# Split the data into a training and test set.
training_data, validation_data, training_labels, validation_labels = train_test_split(breast_cancer_data.data, 
                                              breast_cancer_data.target, train_size = 0.8, random_state=100)


# In[20]:


# Verify if shape of training data and training labels are the same
print(training_data.shape)
print(training_labels.shape)


# In[21]:

# create a function to calculate knn based on different values of n neigbors
def calculateKnn(n_neighbors):
    # Create the knn model.
    knn = KNeighborsRegressor(n_neighbors)

    # Fit the model on the training data.
    knn.fit(training_data, training_labels)
 
    # Calculate accuracy score
    return knn.score(validation_data, validation_labels)
    


# In[22]:

# Search for the best value of k
accuracies=[]
highest_Knn_score = 0
for i in range(1, 101):
    knn_score = calculateKnn(i)
    accuracies.append(knn_score)
    if knn_score > highest_Knn_score:
        highest_Knn_score = knn_score
        best_k_neighbor = i

print('Best value of k: ', best_k_neighbor)
print('Accuracy  score: ', highest_Knn_score)


# In[23]:

#Put the result into a graph
k_list = range(1, 101)
plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("Validation Accuracy") 
plt.title("Breast Cancer Classifier Accuracy")
plt.show()


# <h1> End of Script
