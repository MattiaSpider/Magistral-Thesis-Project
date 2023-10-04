# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

TableNameDATI = "1&2&5-Rev&For-Clean(Voc-1000) - Graph.txt"
TableNameREF = "1&2&3&4&5-Rev&For-AllReference(Voc-1000) - Graph.txt"

pd.set_option('display.max_rows', 10, 'display.max_columns', 10)

# Load the table with full data
Full_Data = pd.read_csv(TableNameDATI, delim_whitespace=True, encoding="iso-8859-1")

# Load the table with full ref data
Full_Data_Ref = pd.read_csv(TableNameREF, delim_whitespace=True, encoding="iso-8859-1")

print(Full_Data)
print()
print(Full_Data_Ref)

print("Dataframe caricati correttamente...")

print()
print("######################################################################")
print()

# Plot Dati Voc & PCE ####################################################################################################################################################################################################################################################################################################################################################################################################################################

# Estrai le iniziali dei nomi dei device + concentrazione + temperatura
Full_Data['Iniziali_C_T'] = Full_Data['DeviceName'].str.extract(r'([A-Za-z\-]+-[0-9A-Za-z]+-[0-9A-Za-z]+)').astype(str) + '°C'

# Plot Voc
fig, ax = plt.subplots(figsize=(15, 30))  # Scambia la dimensione dei subplots

# Color palette standard
color_palette = ['lightcoral'] * 6 + ['red'] * 6 + ['orchid'] * 6 + ['cadetblue'] * 6 + ['blue'] * 6 + ['lightsteelblue'] * 6 + ['green'] * 6 + ['palegreen'] * 6 + ['springgreen'] * 6 

# Color palette modificata
# color_palette = ['red'] * 2 + ['coral'] * 6 + ['lightcoral'] * 6 + ['blue'] * 6 + ['cadetblue'] * 5 + ['lightsteelblue'] * 6 + ['fuchsia'] * 6 + ['orchid'] * 6 + ['mediumpurple'] * 6
# color_palette = ['lightcoral'] * 2 + ['red'] * 6 + ['orchid'] * 6 + ['cadetblue'] * 6 + ['blue'] * 5 + ['lightsteelblue'] * 6 + ['green'] * 6 + ['palegreen'] * 6 + ['springgreen'] * 6 

sns.stripplot(
    y='Iniziali_C_T',  # Scambia l'asse x e y
    x='Voc(mV)',  # Scambia l'asse x e y
    data=Full_Data, 
    size=5, 
    linewidth=0.8, 
    edgecolor='black', 
    palette=color_palette,
    orient='h')  # Imposta l'orientamento orizzontale

sns.boxplot(
    y='Iniziali_C_T',  # Scambia l'asse x e y
    x='Voc(mV)',  # Scambia l'asse x e y
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
    palette=color_palette,
    orient='h')  # Imposta l'orientamento orizzontale

ax.set_yticklabels(ax.get_yticklabels(), fontsize=30)  # Modifica le etichette sull'asse y
ax.tick_params(axis='x', labelsize=30)  # Modifica le etichette sull'asse x

plt.xlabel('Voc(mV)', fontsize=40)  # Modifica le etichette degli assi
plt.ylabel('Device', fontsize=40)  # Modifica le etichette degli assi

ax.set_xlim(900, 1200)  # Modifica il limite dell'asse x

plt.show()

# # Plot PCE
# sns.set_theme(style="ticks")
# fig, ax = plt.subplots(figsize=(30, 20))

# sns.stripplot(
#     x='Iniziali_C_T', 
#     y='PCE(%)', 
#     data=Full_Data, 
#     size=5, 
#     linewidth=0.8, 
#     edgecolor='black', 
#     palette=color_palette)

# sns.boxplot(
#     x='Iniziali_C_T',
#     y='PCE(%)',
#     data=Full_Data,
#     width=0.7,
#     showmeans=True,
#     meanline=True,
#     meanprops={
#         'color': 'black',
#         'linestyle': '-',
#         'linewidth': 2,
#         'marker': 'D',
#         'markersize': 8,
#         'markerfacecolor': 'None',  # Passa la palette come lista
#         'markeredgecolor': 'Black',
#         'markeredgewidth': 1.5, 
#     },
#     palette=color_palette)

# ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=20)
# ax.tick_params(axis='y', labelsize=18)

# plt.title('Box Plot PCE(%)', fontsize=25)
# plt.xlabel('Device', fontsize=20)
# plt.ylabel('PCE(%)', fontsize=20)

# ax.set_ylim(0, 25)

# plt.show()

# Plot Reference Voc & PCE ####################################################################################################################################################################################################################################################################################################################################################################################################################################

# Estrai le iniziali dei nomi dei device + concentrazione + temperatura
Full_Data_Ref['Iniziali_Ref'] = Full_Data_Ref['DeviceName'].str.extract(r'([^-\s]+)')

print(Full_Data_Ref['Iniziali_Ref'])

# Color palette reference
color_palette_ref = ['Grey'] * 5 + ['red'] * 5

# Plot Voc
sns.set_theme(style="ticks")
fig, ax = plt.subplots(figsize=(30, 15))

sns.stripplot(
    x='Iniziali_Ref', 
    y='Voc(mV)', 
    data=Full_Data_Ref, 
    size=5, 
    linewidth=0.8, 
    edgecolor='black', 
    palette=color_palette_ref)

sns.boxplot(
    x='Iniziali_Ref',
    y='Voc(mV)',
    data=Full_Data_Ref,
    width=0.7,
    showmeans=True,
    meanline=True,
    meanprops={
        'color': 'black',
        'linestyle': '-',
        'linewidth': 2,
        'marker': 'D',
        'markersize': 8,
        'markerfacecolor': 'None',  # Passa la palette come lista
        'markeredgecolor': 'Black',
        'markeredgewidth': 1.5, 
    },
    palette=color_palette_ref)

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=30)
ax.tick_params(axis='y', labelsize=30)

# plt.title('Box Plot Voc(mV)', fontsize=25)
plt.xlabel('Device', fontsize=40)
plt.ylabel('Voc(mV)', fontsize=40)

ax.set_ylim(900, 1200)

plt.show()

# # Plot PCE
# sns.set_theme(style="ticks")
# fig, ax = plt.subplots(figsize=(30, 20))

# sns.stripplot(
#     x='Iniziali_Ref', 
#     y='PCE(%)', 
#     data=Full_Data_Ref, 
#     size=5, 
#     linewidth=0.8, 
#     edgecolor='black', 
#     palette=color_palette_ref)

# sns.boxplot(
#     x='Iniziali_Ref',
#     y='PCE(%)',
#     data=Full_Data_Ref,
#     width=0.7,
#     showmeans=True,
#     meanline=True,
#     meanprops={
#         'color': 'black',
#         'linestyle': '-',
#         'linewidth': 2,
#         'marker': 'D',
#         'markersize': 8,
#         'markerfacecolor': 'None',  # Passa la palette come lista
#         'markeredgecolor': 'Black',
#         'markeredgewidth': 1.5, 
#     },
#     palette=color_palette_ref)

# ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=20)
# ax.tick_params(axis='y', labelsize=18)

# plt.title('Box Plot PCE(%)', fontsize=25)
# plt.xlabel('Device', fontsize=20)
# plt.ylabel('PCE(%)', fontsize=20)

# ax.set_ylim(0, 25)

# plt.show()























