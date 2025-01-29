# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 18:49:17 2024

@author: PrajwalBagewadi
"""
#arrays
import numpy as np  
# for reading dataset
import pandas as pd 
# visualization
import matplotlib.pyplot as plt 
# Encoding of dataset
from sklearn.preprocessing import LabelEncoder 
# spliting dataset
from sklearn.model_selection import train_test_split 



from sklearn import tree
#classification library
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC 
#precision matrix 
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay

file=r"C:\Users\bagew\Downloads\archive (1)\data\scores.csv"
data=pd.read_csv(file)
print(f"DataSET:\n{data}")

#categorical data conversion to numeric format
le=LabelEncoder()
data['number']=le.fit_transform(data['number'])
data['age']=le.fit_transform(data['age'])
data['edu']=le.fit_transform(data['edu'])
print(f"data=\n{data}")

#splitting
x=data.iloc[:,:-2]
y=data.iloc[:,-2]
print(f"X=\n{x}")
print(f"y=\n{y}")

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print(f"x_train shape={x_train.shape}")
print(f"x_test shape={x_test.shape}")
print(f"y_train shape={y_train.shape}")
print(f"y_test shape={y_test.shape}")

clf=SVC(kernel='linear',random_state=42)
clf.fit(x_train,y_train)

clf2=DecisionTreeClassifier()
clf2.fit(x_train,y_train)

y_pred=clf.predict(x_test)
print(f"y_pred=\n{y_pred}")
y_pred2=clf2.predict(x_test)
print(f"y_pred2=\n{y_pred2}")

score=accuracy_score(y_test,y_pred)
print(f"accuracy score={score}")
score2=accuracy_score(y_test,y_pred2)
print(f"accuracy score={score2}")

mat=confusion_matrix(y_test,y_pred)
disp=ConfusionMatrixDisplay(confusion_matrix=mat)
disp.plot()
#plt.show()

mat2=confusion_matrix(y_test,y_pred2)
disp2=ConfusionMatrixDisplay(confusion_matrix=mat2)
disp2.plot()
plt.show()
