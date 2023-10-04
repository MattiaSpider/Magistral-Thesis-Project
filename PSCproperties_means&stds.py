# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 21:08:28 2023

@author: utente
"""

import pandas as pd

data = '1&2&5-Rev&For-Clean(Voc-1000)_ML.txt'

# Carica il file CSV in un DataFrame
df = pd.read_csv(data, delim_whitespace=True, encoding="iso-8859-1")

# Estrai il nome del dispositivo dalla prima colonna con condizione
df['DeviceName'] = df['DeviceName'].str.extract(r'([A-Za-z\-]+-[0-9A-Za-z]+-[0-9A-Za-z]+)').astype(str)

print(df['DeviceName'])

# Raggruppa per nome di dispositivo e calcola la media delle prime 4 proprietà arrotondate alla seconda cifra decimale
mean_result = df.groupby('DeviceName').agg({
    'Voc(mV)': lambda x: round(x.mean(), 2),
    'Isc/(mA/cm2)': lambda x: round(x.mean(), 2),
    'FF(%)': lambda x: round(x.mean(), 2),
    'PCE(%)': lambda x: round(x.mean(), 2)
}).reset_index()

# Seleziona il primo valore delle altre colonne all'interno di ciascun gruppo
other_columns = df.columns.difference(['DeviceName', 'Voc(mV)', 'Isc/(mA/cm2)', 'FF(%)', 'PCE(%)'])
mean_result = mean_result.join(df.groupby('DeviceName')[other_columns].first(), on='DeviceName')

# Salva il risultato in un nuovo file CSV
mean_result.to_csv('1&2&5-Rev&For-Clean(Voc-1000)_ML-means.txt', sep=' ', index=False)

# Raggruppa per nome di dispositivo e calcola la std delle prime 4 proprietà arrotondate alla seconda cifra decimale
std_result = df.groupby('DeviceName').agg({
    'Voc(mV)': lambda x: round(x.std(), 2),
    'Isc/(mA/cm2)': lambda x: round(x.std(), 2),
    'FF(%)': lambda x: round(x.std(), 2),
    'PCE(%)': lambda x: round(x.std(), 2)
}).reset_index()

# Calcola la media delle std calcolate
std_mean = std_result[['Voc(mV)', 'Isc/(mA/cm2)', 'FF(%)', 'PCE(%)']].mean()

# Stampa la media delle std su terminale
print("Media delle std:")
print(std_mean)

# Seleziona il primo valore delle altre colonne all'interno di ciascun gruppo
other_columns = df.columns.difference(['DeviceName', 'Voc(mV)', 'Isc/(mA/cm2)', 'FF(%)', 'PCE(%)'])
std_result = std_result.join(df.groupby('DeviceName')[other_columns].first(), on='DeviceName')

# Salva il risultato in un nuovo file CSV
std_result.to_csv('1&2&5-Rev&For-Clean(Voc-1000)_ML-std.txt', sep=' ', index=False)

