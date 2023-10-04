# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 11:38:19 2023

@author: utente
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

TableNameDATI = "7.1-Rev&For-Clean(Voc-1000)-10mml+20T - Graph.txt"

pd.set_option('display.max_rows', 10, 'display.max_columns', 10)

# Load the table with full data
Full_Data = pd.read_csv(TableNameDATI, delim_whitespace=True, encoding="iso-8859-1")

print(Full_Data)
print()

print("Dataframe caricati correttamente...")

print()
print("######################################################################")
print()

# Plot Dati Voc & PCE ####################################################################################################################################################################################################################################################################################################################################################################################################################################

# Estrai le iniziali dei nomi dei device + concentrazione + temperatura
Full_Data['Iniziali_C_T'] = Full_Data['DeviceName'].str.extract(r'([A-Za-z\-]+-[0-9A-Za-z]+-[0-9A-Za-z]+)').astype(str) 

# Plot Voc
fig, ax = plt.subplots(figsize=(10, 8))  # Scambia la dimensione dei subplots

# Color palette standard
color_palette = ['grey'] * 1 + ['cadetblue'] * 1 

sns.stripplot(
    x='Iniziali_C_T',  # Usa 'Iniziali_C_T' sull'asse x
    y='Voc(mV)',  # Usa 'Voc(mV)' sull'asse y
    data=Full_Data, 
    size=5, 
    linewidth=0.8, 
    edgecolor='black', 
    palette=color_palette)  # Non è necessario specificare 'orient' poiché il grafico è verticale

sns.boxplot(
    x='Iniziali_C_T',  # Usa 'Iniziali_C_T' sull'asse x
    y='Voc(mV)',  # Usa 'Voc(mV)' sull'asse y
    data=Full_Data,
    width=0.5,
    showmeans=True,
    meanline=True,
    meanprops={
        'color': 'black',
        'linestyle': '-',
        'linewidth': 2,
        'marker': 'D',
        'markersize': 8,
        'markerfacecolor': 'none',
        'markeredgecolor': 'Black',
        'markeredgewidth': 1.5, 
    },
    palette=color_palette)  # Non è necessario specificare 'orient' poiché il grafico è verticale

# Valore predizione PEACl
voc_value = 1128.67

# Aggiungi una retta orizzontale al grafico per il valore specifico di Voc
plt.axhline(y=voc_value, color='purple', linestyle='--', linewidth=2, label=f'Voc = {voc_value} mV')

# Calcola la confidenza della previsione (AbO) sopra e sotto il valore di Voc
std_dev = 21.33
voc_std_plus = voc_value + std_dev
voc_std_minus = voc_value - std_dev

# Aggiungi linee per la deviazione standard sopra e sotto il valore di Voc
plt.axhline(y=voc_std_plus, color='red', linestyle='--', linewidth=2, label=f'Voc + {std_dev} mV')
plt.axhline(y=voc_std_minus, color='red', linestyle='--', linewidth=2, label=f'Voc - {std_dev} mV')

ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)  # Modifica le etichette sull'asse x
ax.tick_params(axis='y', labelsize=12)  # Modifica le etichette sull'asse y

plt.xlabel('Device', fontsize=12)  # Modifica le etichette degli assi
plt.ylabel('Voc(mV)', fontsize=12)  # Modifica le etichette degli assi

ax.set_ylim(900, 1200)  # Modifica il limite dell'asse y

# Aggiungi la legenda
plt.legend(loc='lower right', fontsize=12)


plt.show()

