# -*- coding: utf-8 -*-
"""
Created on Tue May  1 23:05:59 2018

@author: c_pujain
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import StandardScaler

filename = r'C:\Users\c_pujain\Desktop\abalone.data'

# Importing the dataset
dataset = pd.read_csv(filename)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 8].values


#Encode Categotical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncode_X = LabelEncoder()
X[: , 0] = labelEncode_X.fit_transform(X[:, 0])
oneHotEncder = OneHotEncoder(categorical_features=[0])
X = oneHotEncder.fit_transform(X).toarray()

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train[1:])

clf = tree.DecisionTreeClassifier()

clf = clf.fit(X_train, y_train)

prediction = clf.predict(X_test)

percent = np.subtract(y_test, prediction)

np.count_nonzero(percent == 0)
