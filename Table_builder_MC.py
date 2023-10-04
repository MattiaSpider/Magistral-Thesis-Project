# -*- coding: utf-8 -*-
"""
Created on Thu May 11 15:56:38 2023

@author: Mattia Ragni
Description: questo codice prende la tabella dati in uscita da CalcPCE, rimuove le colonne che non ci 
interessano lasciando quelle del nome del dispositivo, Voc, Jsc, FF e PCE, per poi aggiungere le colonne 
per le variabili di processazione e per le proprietà chimico-fisiche di interesse. 
Fatto ciò recupera dal nome di ogni device il valore di Concentration e temperatura, inserendoli nelle rispettive colonne. 
Infine recupera i valori di tutte le altre variabili dalla tabella in cui sono elencate le proprietà dei passivanti e le inserisce nelle 
colonne corrispondenti ad ogni dispositivo con il suddetto passivante. 
NB1: Bisogna svolgere la pulizia dei dati sul file di CalcPCE prima di creare questa tabella. -> Filtro_PSC
NB2: Bisogna togliere manualmente dal file CalcPCE le celle di Reference.
NB3: è necessario che i valori di concentration e annealing temperature siano scritti correttamente nel nome, se così non 
fosse bisogna correggere il file, è sufficiente un semplice "substitute".

"""

import pandas as pd

TableName = '1&2&3&4-Rev&For-Clean(Voc)_noRef-TEACl.txt'

# Carica il file txt in un DataFrame
df = pd.read_csv(TableName, delim_whitespace=True)

# Rimuovi le ultime 6 colonne, manteniamo solo le prime 5 colonne con nome, Voc, Jsc, FF e PCE
df = df.drop(['area(cm2)', 'Vm(mV)', 'Im(mA)', 'Pm(mW)', 'Rs(ohm.cm2)@1.8V', 'Rsh(kohm.cm2)@0V'], axis=1)

# Aggiungi le colonne "Concentration(mML)" e "Annealing_T(°C)" e quelle per tutte le altre variabili dopo la colonna "PCE(%)"
df.insert(df.columns.get_loc("PCE(%)")+1, "Concentration(mML)", "")
df.insert(df.columns.get_loc("PCE(%)")+2, "Annealing_T(C)", "")
df.insert(df.columns.get_loc("PCE(%)")+3, "Molecular_Weight", "")
df.insert(df.columns.get_loc("PCE(%)")+4, "Hydrogen_Bond_Donor_Count", "")
df.insert(df.columns.get_loc("PCE(%)")+5, "Hydrogen_Bond_Acceptor_Count", "")
df.insert(df.columns.get_loc("PCE(%)")+6, "Rotatable_Bond_Count", "")
df.insert(df.columns.get_loc("PCE(%)")+7, "Rigid_Bond_Count", "")
df.insert(df.columns.get_loc("PCE(%)")+8, "Topological_Polar_Surface_Area", "")
df.insert(df.columns.get_loc("PCE(%)")+9, "Complexity", "")
df.insert(df.columns.get_loc("PCE(%)")+10, "Heavy_Atom_Count", "")
df.insert(df.columns.get_loc("PCE(%)")+11, "Halide_fraction", "")
df.insert(df.columns.get_loc("PCE(%)")+12, "Heteroatom_carbon_ratio", "")
df.insert(df.columns.get_loc("PCE(%)")+13, "Heteroatom_count", "")
df.insert(df.columns.get_loc("PCE(%)")+14, "H_Count", "")
df.insert(df.columns.get_loc("PCE(%)")+15, "C_Count", "")
df.insert(df.columns.get_loc("PCE(%)")+16, "Cl_Count", "")
df.insert(df.columns.get_loc("PCE(%)")+17, "Br_Count", "")
df.insert(df.columns.get_loc("PCE(%)")+18, "I_Count", "")


# estrai il valore della concentration dal nome della prima colonna
df['Concentration(mML)'] = df['DeviceName'].str.extract('(\d+)mML', expand=False).fillna(0).astype(int)

# estrai il valore della temperatura dal nome della prima colonna
df['Annealing_T(C)'] = df['DeviceName'].str.extract('(\d+)T', expand=False).fillna(0).astype(int)

df2 = pd.read_csv('Proprietà_cationi_NEW.txt', delim_whitespace=True)

print(df)
print(df2)

def update_df(name):
    # Trova le righe contenenti il nome in df1
    mask = df['DeviceName'].str.contains(name)

    # Ottieni i valori corrispondenti da df2
    mol_weight = df2.loc[df2['Name'] == name, 'Molecular_Weight'].values[0]
    hbd_count = df2.loc[df2['Name'] == name, 'Hydrogen_Bond_Donor_Count'].values[0]
    hba_count = df2.loc[df2['Name'] == name, 'Hydrogen_Bond_Acceptor_Count'].values[0]
    rot_bond_count = df2.loc[df2['Name'] == name, 'Rotatable_Bond_Count'].values[0]
    tpsa = df2.loc[df2['Name'] == name, 'Topological_Polar_Surface_Area'].values[0]
    heavy_atom_count = df2.loc[df2['Name'] == name, 'Heavy_Atom_Count'].values[0]
    complexity = df2.loc[df2['Name'] == name, 'Complexity'].values[0]
    
    H_Count = df2.loc[df2['Name'] == name, 'H_Count'].values[0]
    C_Count = df2.loc[df2['Name'] == name, 'C_Count'].values[0]
    I_Count = df2.loc[df2['Name'] == name, 'I_Count'].values[0]
    Br_Count = df2.loc[df2['Name'] == name, 'Br_Count'].values[0]
    Cl_Count = df2.loc[df2['Name'] == name, 'Cl_Count'].values[0]
    
    Hf_Count = df2.loc[df2['Name'] == name, 'Halide_fraction'].values[0]
    HACf_Count = df2.loc[df2['Name'] == name, 'Heteroatom_carbon_ratio'].values[0]
    HA_Count = df2.loc[df2['Name'] == name, 'Heteroatom_count'].values[0]
    rig_bond_count = df2.loc[df2['Name'] == name, 'Rigid_Bond_Count'].values[0]
    
    
    # Inserisci i valori nella tabella df1
    df.loc[mask, 'Molecular_Weight'] = mol_weight
    df.loc[mask, 'Hydrogen_Bond_Donor_Count'] = hbd_count
    df.loc[mask, 'Hydrogen_Bond_Acceptor_Count'] = hba_count
    df.loc[mask, 'Rotatable_Bond_Count'] = rot_bond_count
    df.loc[mask, 'Topological_Polar_Surface_Area'] = tpsa
    df.loc[mask, 'Heavy_Atom_Count'] = heavy_atom_count
    df.loc[mask, 'Complexity'] = complexity
    
    df.loc[mask, 'H_Count'] = H_Count
    df.loc[mask, 'C_Count'] = C_Count
    df.loc[mask, 'I_Count'] = I_Count
    df.loc[mask, 'Br_Count'] = Br_Count
    df.loc[mask, 'Cl_Count'] = Cl_Count
    
    df.loc[mask, 'Halide_fraction'] = Hf_Count
    df.loc[mask, 'Heteroatom_carbon_ratio'] = HACf_Count
    df.loc[mask, 'Heteroatom_count'] = HA_Count
    df.loc[mask, 'Rigid_Bond_Count'] = rig_bond_count

# Lista nomi passivanti
nomi = ['MEPEAI', 'MEPEABr', 'MEPEACl', 'n-BAI', 'iso-BAI', 'n-OAI', 'TEACl', 'TMABr', 'AVAI', 'BBr', 'HBr', 'OATsO']

# Applica la funzione ad ogni nome nella lista
for nome in nomi:
    update_df(nome)
    
print(df)    

# Salva il DataFrame modificato
df.to_csv(TableName.replace('.txt', '_ML.txt'), sep=" ", index=False)




