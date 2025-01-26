#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 18:15:47 2024

@author: thomasroujou
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Charger les données
data = pd.read_csv("/Users/thomasroujou/Desktop/Task 3 and 4_loan_Data.csv")

# Prétraitement des données pour s'assurer que les colonnes pertinentes sont correctes
data = data[["fico_score", "default"]].dropna()  # Utiliser uniquement les colonnes nécessaires

# Quantification basée sur l'erreur quadratique moyenne (MSE)
def quantification_fico_mse(data, colonne_fico, colonne_defaut, n_buckets):
    """
    Quantification des scores FICO en minimisant l'erreur quadratique moyenne (MSE).
    """
    fico_scores = data[colonne_fico].sort_values().values
    bornes = np.percentile(fico_scores, np.linspace(0, 100, n_buckets + 1))
    
    # Attribution des ratings
    ratings = pd.cut(data[colonne_fico], bins=bornes, labels=range(1, n_buckets + 1), include_lowest=True)
    return bornes, ratings

# Quantification basée sur la log-vraisemblance
def bornes_log_vraisemblance(data, colonne_fico, colonne_defaut, n_buckets):
    """
    Quantification des scores FICO en maximisant la log-vraisemblance.
    """
    fico_scores = data[colonne_fico].sort_values().values
    defaults = data[colonne_defaut].values
    
    # Fonction pour calculer la log-vraisemblance
    def calcul_log_vraisemblance(bornes):
        bornes = np.sort(bornes)
        ll = 0
        for i in range(len(bornes) - 1):
            masque_tranche = (fico_scores >= bornes[i]) & (fico_scores < bornes[i + 1])
            n_i = masque_tranche.sum()
            k_i = defaults[masque_tranche].sum()
            if n_i > 0:
                p_i = k_i / n_i
                ll += k_i * np.log(p_i + 1e-9) + (n_i - k_i) * np.log(1 - p_i + 1e-9)
        return -ll  # Négatif car nous minimisons

    # Initialisation des bornes
    bornes_initiales = np.percentile(fico_scores, np.linspace(0, 100, n_buckets + 1))[1:-1]
    resultat = minimize(calcul_log_vraisemblance, bornes_initiales, method="Powell")
    
    bornes = np.concatenate(([fico_scores.min()], resultat.x, [fico_scores.max()]))
    bornes.sort()
    
    # Attribution des ratings
    ratings = pd.cut(data[colonne_fico], bins=bornes, labels=range(1, n_buckets + 1), include_lowest=True)
    return bornes, ratings

# Nombre de tranches (buckets)
n_buckets = 5

# Quantification basée sur le MSE
bornes_mse, ratings_mse = quantification_fico_mse(data, "fico_score", "default", n_buckets)
print("Bornes (MSE) :", bornes_mse)
print("Carte des ratings (MSE) :", ratings_mse.value_counts().sort_index())

# Quantification basée sur la log-vraisemblance
bornes_ll, ratings_ll = bornes_log_vraisemblance(data, "fico_score", "default", n_buckets)
print("Bornes (Log-vraisemblance) :", bornes_ll)
print("Carte des ratings (Log-vraisemblance) :", ratings_ll.value_counts().sort_index())
