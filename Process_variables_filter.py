# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 19:27:56 2023

@author: utente
"""

from pandas import read_csv

# Leggo il file
data_file = '1&2&5-Rev&For-Clean(Voc)-10mml+20T - Graph.txt'
Full_Data = read_csv(data_file, delimiter=" ")

# Filtro le righe che soddisfano le condizioni desiderate
Full_Data = Full_Data[(Full_Data['Annealing_T(C)'] == 20) & (Full_Data['Concentration(mml)'] == 10)]

# Salvo il dataframe risultante come file .txt con lo stesso nome del file originale
Full_Data.to_csv(data_file, index=False, sep=' ')

print(f"Il dataframe è stato salvato come {data_file}")
