# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 14:25:25 2018

@author: c_pujain
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import tree

filename = r'C:\Workspace\ML\ML-Projects\Iris\iris.data'

# Importing the dataset
dataset = pd.read_csv(filename)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4:5].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 5)



# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

prediction = clf.predict(X_test).tolist()
y_test = y_test.tolist()

total = len(y_test)
count = 0
for i in range(total):
    if prediction[i] == y_test[i][0]:
        print(prediction[i], y_test[i][0])
        count += 1