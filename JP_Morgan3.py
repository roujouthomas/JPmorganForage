#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 17:09:03 2024

@author: thomasroujou
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score


data = pd.read_csv("/Users/thomasroujou/Desktop/Task 3 and 4_loan_Data.csv")


colonnes_utiles = [
    "credit_lines_outstanding", 
    "loan_amt_outstanding", 
    "total_debt_outstanding", 
    "income", 
    "years_employed", 
    "fico_score", 
    "default"
]
data = data[colonnes_utiles]


X = data.drop("default", axis=1)  # Caractéristiques
y = data["default"]  # Variable cible


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#modèle foret aléatoire
modele = RandomForestClassifier(random_state=42)
modele.fit(X_train, y_train)


y_pred = modele.predict(X_test)
y_proba = modele.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
print(f"AUC : {roc_auc_score(y_test, y_proba):.2f}")


def perte_attendue(caracteristiques, modele, taux_recouvrement=0.1):
    """
    Calcule la perte attendue pour un prêt.

    Arguments :
    - caracteristiques : dict, caractéristiques du prêt (ex. montant du prêt, revenus, etc.)
    - modele : modèle entraîné pour prédire la probabilité de défaut
    - taux_recouvrement : float, taux de recouvrement (10 % par défaut)

    Retourne :
    - perte attendue : float
    """
    df_caracteristiques = pd.DataFrame([caracteristiques])
    proba_defaut = modele.predict_proba(df_caracteristiques)[:, 1][0]
    montant_pret = caracteristiques["loan_amt_outstanding"]
    perte = montant_pret * (1 - taux_recouvrement) * proba_defaut
    return perte


emprunteur = {
    "credit_lines_outstanding": 3,
    "loan_amt_outstanding": 20000,
    "total_debt_outstanding": 50000,
    "income": 60000,
    "years_employed": 8,
    "fico_score": 678
}
print(f"expected_los : ${perte_attendue(emprunteur, modele):,.2f}")
