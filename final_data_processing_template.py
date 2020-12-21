# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 11:38:53 2020

@author: Fly to the sky
"""

#Chargement des packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import missingno as msno

#Chargements des données
dataset = pd.read_csv("C:/Users/utilisateur\Desktop/projects/ML/dataset.csv")
dataset.drop(columns=['Unnamed: 0'], inplace=True)
dataset.Sex = dataset.Sex.map({'m': 1, 'f':2})
dataset.Category = dataset.Category.map({'0=Blood Donor': 1, '1=Hepatitis': 0, '2=Fibrosis': 0, '3=Cirrhosis': 0, '0s=suspect Blood Donor': 0})


#Découpage des données par classe
blood_donor = dataset.iloc[:533, :].values
suspect_blood_donor = dataset.iloc[533:540, :].values
hepatitis = dataset.iloc[540:564, :].values
fibrosis = dataset.iloc[564:585, :].values
cirrhosis = dataset.iloc[585:, :].values


#Traitements des données manquantes par classe
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

imputer = imputer.fit(blood_donor[:,3:])
blood_donor[:,3:] = imputer.transform(blood_donor[:,3:])

imputer = imputer.fit(hepatitis[:,3:])
hepatitis[:,3:] = imputer.transform(hepatitis[:,3:])

imputer = imputer.fit(fibrosis[:,3:])
fibrosis[:,3:] = imputer.transform(fibrosis[:,3:])

imputer = imputer.fit(cirrhosis[:,3:])
cirrhosis[:,3:] = imputer.transform(cirrhosis[:,3:])

#fusion des tableaux 
z = np.concatenate((blood_donor, suspect_blood_donor, hepatitis, fibrosis, cirrhosis), axis=0)
x = z[:, 1:]
y = z[:, 0]

#découpage et apprentissage

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size = 0.25)
classifier = LogisticRegression(random_state = 0, solver = 'liblinear')
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
score = classifier.score(x_test, y_test)
cm = confusion_matrix(y_test, y_pred)
print(cm, 'score: ', score)

fpr, tpr, threshold = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()