# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 01:41:18 2020

@author: Vaibhav
"""


#Importing libraries:

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



#Importing Dataset:

df = pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)

#Data Preprocessing:

print(data.head())

print(data.tail())

data.shape

data.isna().sum()



#Data Cleaning 

import re

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

corpus = []

for i in range(0,1000):
    review = df['Review'][i]
    review = re.sub('[^a-zA-Z] ',' ',review)
    review = review.lower()
    
    review = review.split()
    
    all_stopwords = stopwords.words('english')
    
    rm_words=['not','no','nor']
    
    for word in list(rm_words):
        if word in all_stopwords:
            all_stopwords.remove(word)
            
    ps = PorterStemmer()
    
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    
    review = ' '.join(review)
    corpus.append(review)
    
print(all_stopwords)





#Creating Bag of Words Model:

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
# the CountVectorizer class take one parameter 'max_features' which helps to select the number of columns/features in x
x = cv.fit_transform(corpus).toarray()
y = df.iloc[:,-1].values
len(x[0])
print(x)



##SVM Model:

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.svm import SVC
classifier = SVC(kernel='linear',random_state=0)
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

                        #Plotting Confusion Matrix and Checking Accuracy Score

from sklearn.metrics import plot_confusion_matrix,accuracy_score
plot_confusion_matrix(classifier,x_test,y_test)
acc = accuracy_score(y_test,y_pred)
print(acc)




##KNN Model:
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski',p=2)
knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)
y_pred

                         #Plotting Confusion Matrix and Checking Accuracy Score

from sklearn.metrics import plot_confusion_matrix,accuracy_score
plot_confusion_matrix(knn,x_test,y_test,cmap = 'Blues')
acc = accuracy_score(y_test,y_pred)
print(acc)