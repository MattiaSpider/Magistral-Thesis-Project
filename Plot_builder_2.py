# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

TableNameDATI = "1&2&5-Rev&For-Clean(FF-60)-10mml+20T - Graph.txt"
# TableNameREF = "1&2&5-Rev&For-AllReference(Voc-1000) - Graph.txt"

pd.set_option('display.max_rows', 10, 'display.max_columns', 10)

# Load the table with full data
Full_Data = pd.read_csv(TableNameDATI, delim_whitespace=True, encoding="iso-8859-1")

# Load the table with full ref data
# Full_Data_Ref = pd.read_csv(TableNameREF, delim_whitespace=True, encoding="iso-8859-1")

print(Full_Data)
print()
# print(Full_Data_Ref)

print("Dataframe caricati correttamente...")

print()
print("######################################################################")
print()

# Plot Dati Voc & PCE ####################################################################################################################################################################################################################################################################################################################################################################################################################################

# Estrai le iniziali dei nomi dei device + concentrazione + temperatura
Full_Data['Iniziali_C_T'] = Full_Data['DeviceName'].str.extract(r'([A-Za-z\-]+-[0-9A-Za-z]+-[0-9A-Za-z]+)').astype(str) + '°C'

# Plot Voc
fig, ax = plt.subplots(figsize=(30, 20))

# # Color palette standard
# color_palette = ['darkred'] * 6 + ['red'] * 6 + ['lightcoral'] * 6 + ['darkblue'] * 6 + ['blue'] * 6 + ['lightsteelblue'] * 6 + ['green'] * 6 + ['palegreen'] * 6 + ['indigo'] * 6 + ['darkviolet'] * 6 + ['orchid'] * 6

# # Color palette modificata
# color_palette = ['red'] * 2 + ['coral'] * 6 + ['lightcoral'] * 6 + ['blue'] * 6 + ['cadetblue'] * 5 + ['lightsteelblue'] * 6 + ['fuchsia'] * 6 + ['orchid'] * 6 + ['mediumpurple'] * 6

# Creazione di una palette automatica
# color_palette = sns.color_palette("bright", n_colors=len(Full_Data['Iniziali_C_T'].unique()))
color_palette = ['lightcoral'] * 1 + ['red'] * 1 + ['orchid'] * 1 + ['cadetblue'] * 1 + ['blue'] * 1 + ['lightsteelblue'] * 1 + ['green'] * 1 + ['palegreen'] * 1 + ['springgreen'] * 1 

sns.stripplot(
    x='Iniziali_C_T', 
    y='FF(%)', 
    data=Full_Data, 
    size=5, 
    linewidth=0.8, 
    edgecolor='black', 
    palette=color_palette)

sns.boxplot(
    x='Iniziali_C_T',
    y='FF(%)',
    data=Full_Data,
    width=0.7,
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
    palette=color_palette)

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=20)
ax.tick_params(axis='y', labelsize=18)

plt.title('Box Plot FF (%)', fontsize=25)
plt.xlabel('Device', fontsize=20)
plt.ylabel('FF (%))', fontsize=20)

ax.set_ylim(50, 90)

plt.show()






















