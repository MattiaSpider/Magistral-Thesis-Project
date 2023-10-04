# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 19:54:38 2023

@author: utente
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

TableName = "1&2&5-Rev&For-Clean(Voc-1000)_ML.txt"
pd.set_option('display.max_rows', 10, 'display.max_columns', 10)

# The molecular features & processing conditions are loaded as X
# All
# ProcessingData_cols = ["DeviceName", "Concentration(mML)", "Annealing_T(C)", "Molecular_Weight", "Rigid_Bond_Count", "Rotatable_Bond_Count", "Topological_Polar_Surface_Area", "Hydrogen_Bond_Donor_Count", "Hydrogen_Bond_Acceptor_Count", "Heavy_Atom_Count", "Complexity", "H_Count", "C_Count", "Cl_Count", "Br_Count", "I_Count", "Halide_fraction", "Heteroatom_carbon_ratio", "Heteroatom_count"]
# SignificativeOLD
# ProcessingData_cols = ["DeviceName", "Concentration(mML)", "Annealing_T(C)", "Molecular_Weight", "Rigid_Bond_Count", "Rotatable_Bond_Count", "Topological_Polar_Surface_Area", "Complexity", "Hydrogen_Bond_Donor_Count", "Hydrogen_Bond_Acceptor_Count", "H_Count", "C_Count", "Cl_Count", "Br_Count", "I_Count", "Halide_fraction", "Heteroatom_carbon_ratio"]
# SignificativeNEW (no TPSA & HBDC & HBAC)
# ProcessingData_cols = ["DeviceName", "Concentration(mML)", "Annealing_T(C)", "Molecular_Weight", "Rigid_Bond_Count", "Rotatable_Bond_Count", "Complexity", "H_Count", "C_Count", "Cl_Count", "Br_Count", "I_Count", "Halide_fraction", "Heteroatom_carbon_ratio"]
# SignificativeNEW (no TPSA & HBDC & HBAC & H_count & C_count)
ProcessingData_cols = ["DeviceName", "Concentration(mML)", "Annealing_T(C)", "Molecular_Weight", "Rigid_Bond_Count", "Rotatable_Bond_Count", "Complexity", "Cl_Count", "Br_Count", "I_Count", "Halide_fraction", "Heteroatom_carbon_ratio"]
# SignificativeNEW (no TPSA & HBDC & HBAC & H_count & C_count & Annealing) 
# ProcessingData_cols = ["DeviceName", "Concentration(mML)", "Molecular_Weight", "Rigid_Bond_Count", "Rotatable_Bond_Count", "Complexity", "Cl_Count", "Br_Count", "I_Count", "Halide_fraction", "Heteroatom_carbon_ratio"]
# SignificativeNEW (no TPSA & HBDC & HBAC & H_count & C_count & Annealing & Concentration) 
# ProcessingData_cols = ["DeviceName", "Molecular_Weight", "Rigid_Bond_Count", "Rotatable_Bond_Count", "Complexity", "Cl_Count", "Br_Count", "I_Count", "Halide_fraction", "Heteroatom_carbon_ratio"]
# No Atom_count
# ProcessingData_cols = ["DeviceName", "Concentration(mML)", "Annealing_T(C)", "Molecular_Weight", "Rigid_Bond_Count", "Rotatable_Bond_Count", "Topological_Polar_Surface_Area", "Complexity", "Hydrogen_Bond_Donor_Count", "Hydrogen_Bond_Acceptor_Count", "Halide_fraction", "Heteroatom_carbon_ratio"]

# Load the table with full data
Full_Data = pd.read_csv(TableName, delim_whitespace=True, encoding="iso-8859-1")

# # Check if the data is loaded and separated correctly
# print(Full_Data)

# Creo il dataframe con solo i valori di processazione e proprietà fisico-chimiche (input)
X = Full_Data[ProcessingData_cols]

# Creo il dataframe con solo i valori di Voc (output)
Y_Voc = Full_Data["Voc(mV)"]

# # Check if the data is loaded and separated correctly
# pd.set_option('display.max_rows', 10, 'display.max_columns', 5)
# print(X, Y_Voc)

print()

print("Dataframe caricati correttamente...")

print()
print("######################################################################")
print()

########################################################################################################################################################################################################

# Ottengo valori Voc legati ai nomi dei device
Y_Voc = Full_Data[["DeviceName", "Voc(mV)"]]

# Tutte le sigle in DeviceName
all_device_initials = [['MEPEAI-5mML-20T',
                        'MEPEAI-5mML-100T',
                        'MEPEAI-10mML-20T',
                        'MEPEAI-10mML-100T',
                        'MEPEAI-15mML-20T',
                        'MEPEAI-15mML-100T'],
                       ['MEPEABr-5mML-20T',
                        'MEPEABr-5mML-100T',
                        'MEPEABr-10mML-20T',
                        'MEPEABr-10mML-100T',
                        'MEPEABr-15mML-20T',
                        'MEPEABr-15mML-100T'],
                       ['MEPEACl-5mML-20T',
                        'MEPEACl-5mML-100T',
                        'MEPEACl-10mML-20T',
                        'MEPEACl-10mML-100T',
                        'MEPEACl-15mML-20T',
                        'MEPEACl-15mML-100T'],
                       ['n-BAI-5mML-20T',
                        'n-BAI-5mML-100T',
                        'n-BAI-10mML-20T',
                        'n-BAI-10mML-100T',
                        'n-BAI-15mML-20T',
                        'n-BAI-15mML-100T'],
                       ['iso-BAI-5mML-20T',
                        'iso-BAI-5mML-100T',
                        'iso-BAI-10mML-20T',
                        'iso-BAI-10mML-100T',
                        'iso-BAI-15mML-20T',
                        'iso-BAI-15mML-100T'],
                       ['n-OAI-5mML-20T',
                        'n-OAI-5mML-100T',
                        'n-OAI-10mML-20T',
                        'n-OAI-10mML-100T',
                        'n-OAI-15mML-20T',
                        'n-OAI-15mML-100T'],
                       ['BBr-5mML-20T',
                        'BBr-5mML-100T',
                        'BBr-10mML-20T',
                        'BBr-10mML-100T',
                        'BBr-15mML-20T',
                        'BBr-15mML-100T'],
                       ['HBr-5mML-20T',
                        'HBr-5mML-100T',
                        'HBr-10mML-20T',
                        'HBr-10mML-100T',
                        'HBr-15mML-20T',
                        'HBr-15mML-100T'],
                       ['OATsO-5mML-20T',
                        'OATsO-5mML-100T',
                        'OATsO-10mML-20T',
                        'OATsO-10mML-100T',
                        'OATsO-15mML-20T',
                        'OATsO-15mML-100T']]

#lista in cui verranno inseriti gli errori medi dei modelli per ogni ciclo 
rf_errors = []


#lista in cui verranno inseriti le std medie dei modelli per ogni ciclo 
rf_stds = []

#lista in cui verranno inseriti le combinazioni di best iperparametri per ogni ciclo
rf_bestparameters = []


# Validazione all but one per estrarre RF iperparametri con score collegato
for device_initials in all_device_initials:
    
    # Estrai l'iniziale comune dalla prima stringa nella lista
    import re 
    common_initial = re.split(r'-\d', device_initials[0])[0]
        
    # Creo train set con n-1 cationi e test set con n-esimo catione
    X_train = X[~X['DeviceName'].str.startswith(tuple(device_initials))]
    X_test = X[X['DeviceName'].str.startswith(tuple(device_initials))]
    Y_Voc_train = Y_Voc[~Y_Voc['DeviceName'].str.startswith(tuple(device_initials))]
    Y_Voc_test = Y_Voc[Y_Voc['DeviceName'].str.startswith(tuple(device_initials))]

    print("Train con n-1 cationi e test con n-esimo catione pronti...")
    print()
    print("######################################################################")
    print()

    # Ciclo per calcolare la media dei valori sperimentali per i 6 device nella lista corrente
    experimental_values_mean = []  # Lista per salvare le medie dei valori sperimentali
    for device_name in device_initials:
        # Filtra i valori Voc corrispondenti al nome device_name nella colonna DeviceName
        values = Y_Voc_test[Y_Voc_test["DeviceName"].str.contains(device_name)]["Voc(mV)"].values
        
        # print(values)
        
        # Calcola la media dei valori sperimentali
        values_mean = np.mean(values)
        experimental_values_mean.append(values_mean)
    
    print(experimental_values_mean)
    
    print()
    print("Medie valori sperimentali device con n-esimo catione calcolate...")
    print()
    print("######################################################################")
    print()
    
    # Tolgo colonna dei nomi
    Y_Voc_train = Y_Voc_train.drop("DeviceName", axis=1)
    Y_Voc_test = Y_Voc_test.drop("DeviceName", axis=1)
    
    # Trasformo da colonna a riga 
    Y_Voc_train = Y_Voc_train.values.ravel() 
    Y_Voc_test = Y_Voc_test.values.ravel() 
    
    # print(X_train)
    # print(X_test)
       
    ########################################################################################################################################################################################################
    
    #Standardizzazione valori di input con Standard Scaler
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X_noname = X.drop("DeviceName", axis=1)
    X_train_noname = X_train.drop("DeviceName", axis=1)
    X_test_noname = X_test.drop("DeviceName", axis=1)
    
    #Standardizzazione
    scaler.fit(X_noname)
    X_train_stand = scaler.transform(X_train_noname)
    X_train_stand_no = scaler.transform(X_train_noname) #rinominazione per lo shuffle
    X_test_stand = scaler.transform(X_test_noname) 
    
    print("Standardizzazione dei valori di input svolta...")
    print()
    print("######################################################################")
    print()
    
    # #No standardizzazione
    # X_train_stand = X_train_noname
    # X_test_stand = X_test_noname
    
    ########################################################################################################################################################################################################

    # Random Forest
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV
    
    def rfr_model(X, y):
    
        # Perform Grid-Search
        gsc = GridSearchCV(
              estimator=RandomForestRegressor(random_state=42),
              param_grid={
                  'max_depth': range(1,5),
                  'n_estimators': (5,10,20,30,40,50,60,70,80,90,100,500,1000),
              },
              cv=5, 
              scoring= 'neg_mean_squared_error', 
              verbose=0,
              n_jobs=-1)
        
        grid_result = gsc.fit(X, y)
        best_params = grid_result.best_params_
        rf_regressor = RandomForestRegressor(**best_params)  # Utilizza i migliori parametri trovati
            
        return rf_regressor, best_params
    
    rf_regressor, best_params = rfr_model(X_train_stand, Y_Voc_train)
    print('The best rf parameters: ', best_params)
    
    # rf_regressor = RandomForestRegressor(max_depth = 4,
    #                                       n_estimators=20,
    #                                       random_state=42)
    
    # Fit to the training set
    rf_regressor.fit(X_train_stand, Y_Voc_train)
    # Perform predictions on both training and test sets
    Y_Voc_train_RF = rf_regressor.predict(X_train_stand)
    Y_Voc_test_RF = rf_regressor.predict(X_test_stand)
    
    print("Random Forest trainato...")
    print()
    print("######################################################################")
    print()
   
    ########################################################################################################################################################################################################

    # Otteniamo X_test_stand con solo 6 righe, una per ogni combinazione della lista corrente
    X_test_6 = pd.DataFrame(columns=X.columns)  
    for initial in device_initials:
        device_data = X[X['DeviceName'].str.startswith(initial)]
        if len(device_data) > 0:
            row = device_data.iloc[0]  # Prendi la prima riga del gruppo
            X_test_6 = X_test_6.append(row, ignore_index=True)
            
    # Applico stesso scaler anche a X_test_6
    X_test_6_noname = X_test_6.drop("DeviceName", axis=1)
    X_test_6_stand = scaler.transform(X_test_6_noname) 
    
    # No standardizzazione 
    # X_test_6_stand = X_test_6.drop("DeviceName", axis=1)
        
    # Otteniamo le medie dei valori sperimentali per ogni combinazione contenuta nella lista corrente
    Y_Voc_test_n_mean = []
    for device in device_initials:
        device_data = Y_Voc[Y_Voc['DeviceName'].str.startswith(device)]
        mean_voc_value = np.mean(device_data['Voc(mV)'])
        Y_Voc_test_n_mean.append(mean_voc_value)

    # print(Y_Voc_test_n_mean)    

    # Ottieni predizioni per RF sulle 6 combinazioni dell'n-esimo catione
    Y_Voc_test_RF = rf_regressor.predict(X_test_6_stand)

    # Ricaviamo l'RMSE e la std
    from sklearn import metrics
    rf_score = np.sqrt(metrics.mean_squared_error(Y_Voc_test_n_mean, Y_Voc_test_RF))
   
    rf_std = np.std(Y_Voc_test_n_mean - Y_Voc_test_RF)
    
    #inserisco gli score ottenuti nella lista
    rf_errors.append(rf_score)  
     
    print(rf_errors)
    
    #inserisco le std ottenute nella lista
    rf_stds.append(rf_std)  
  
    print(rf_stds)
    
    #inserisco le i best parameters ottenuti nella lista
    rf_bestparameters.append(best_params)  
    
    print()
    print("Random forest All but One con", common_initial, "svolto")
    print()
    print("-------------------------------------------------------------------------")
    print()

# Controllo coerenza con quanto ottenuto nell'AbO con tutti i modelli
rf_mean_error = np.mean(rf_errors) 
rf_mean_std = np.mean(rf_stds)

print('Mean RMSE values: ' + str(rf_mean_error))
print('Mean Std values: ' + str(rf_mean_std))

# Visualizza risultato

print('RF All but One validation RMSE scores for each cicle best parameters: ' , rf_errors)
print('RF All but One validation std scores for each cicle best parameters: ' , rf_stds)
print('RF All but One validation std scores for each cicle best parameters: ' , rf_bestparameters)

fig, ax = plt.subplots(figsize=(8, 6))

# Creazione delle etichette basate sui valori degli iperparametri
labels = []
label_counts = {}  # Contatore per tener traccia delle etichette duplicate

for params in rf_bestparameters:
    label = f"{params['max_depth']}, {params['n_estimators']}"
    
    if label in label_counts:
        label_counts[label] += 1
        labels.append(f"{label} ({label_counts[label]})")
    else:
        label_counts[label] = 1
        labels.append(label)
        
# Creazione delle barre con alpha e capsize
ax.bar(labels, rf_errors, yerr=rf_stds, alpha=0.7, capsize=5)

# Aggiungi alcune proprietà di formattazione
ax.set_ylabel('RMSE (mV)')
ax.set_title('RMSE scores for RF best hyperparameters combinations')
ax.set_xlabel('Max Depth - Num Estimators')
plt.xticks(rotation=90)

plt.show()