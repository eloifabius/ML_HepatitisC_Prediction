#Préparation des données

#Chargement des packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import missingno as msno

#Chargements des données
dataset = pd.read_csv("hcvdat0.csv")

#Visualisation des données manquantes
msno.matrix(dataset)

x = dataset.iloc[:, 2:].values
y = dataset.iloc[:, 1].values

#Traitements des données manquantes
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

imputer = imputer.fit(x[:, 2:-1])
x[:, 2:-1] = imputer.transform(x[:, 2:-1])

#concatener les deux tableaux pour généré un fichier csv avec les donées complétées
z = np.c_[y, x]

#Généré fichier csv
entetes = [
     u'Category',
     u'Age',
     u'Sex',
     u'ALB',
     u'ALP',
     u'ALT',
     u'AST',
     u'BIL',
     u'CHE',
     u'CHOL',
     u'CREA',
     u'GGT',
     u'PROT'
]

f = open('dataset_final.csv', 'w')
ligneEntete = ";".join(entetes) + "\n"
f.write(ligneEntete)
for val in z:
    taille = len(val)
    for i in range(taille):
        val[i] = str(val[i])
        
    ligne = ";".join(val) + "\n"
    f.write(ligne)
f.close()    