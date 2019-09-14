# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 12:49:12 2019

@author: Sriharsha Komera
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the dataset
path='F:\\Krish\\\ANN\\Churn_Modelling.csv'
dataset=pd.read_csv(path)
X=dataset.iloc[:,3:13]
y=dataset.iloc[:,13]

#creating dummy variables
geography=pd.get_dummies(X['Geography'],drop_first=True)
gender=pd.get_dummies(X['Gender'],drop_first=True)

#Concatinating
X=pd.concat([X,geography,gender],axis=1)

#Dropping the existing column
X=X.drop(['Geography','Gender'],axis=1)

#splitting the data into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense
#from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout

#Initialising the ANN
classifier=Sequential()

#Adding the Input layer and the first hidden layer
classifier.add(Dense(units=6, kernel_initializer= 'he_uniform', activation='relu',input_dim=11))

#Adding the second hidden layer
classifier.add(Dense(units=6, kernel_initializer= 'he_uniform', activation='relu'))

#Adding the Output layer
classifier.add(Dense(units=1, kernel_initializer= 'glorot_uniform', activation='sigmoid'))

classifier.summary()

#compilying the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Fitting the ANN to the training set
model_history=classifier.fit(X_train,y_train,validation_split=0.33,batch_size=10,nb_epoch=100)

#predecting the test set
y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

# Accuracy score
from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)

#summarise history of loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()








