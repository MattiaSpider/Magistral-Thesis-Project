# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 22:38:22 2023

@author: Mattia Ragni
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

TableName = "1&2&5-Rev&For-Clean(Voc-1000)_ML-means.txt"
pd.set_option('display.max_rows', 10, 'display.max_columns', 10)

# The molecular features & processing conditions are loaded as X
# All
ProcessingData_cols = ["DeviceName", "Concentration(mML)", "Annealing_T(C)", "Molecular_Weight", "Rigid_Bond_Count", "Rotatable_Bond_Count", "Topological_Polar_Surface_Area", "Hydrogen_Bond_Donor_Count", "Hydrogen_Bond_Acceptor_Count", "Heavy_Atom_Count", "Complexity", "H_Count", "C_Count", "Cl_Count", "Br_Count", "I_Count", "Halide_fraction", "Heteroatom_carbon_ratio", "Heteroatom_count"]
# SignificativeOLD (no heteroatom count)
# ProcessingData_cols = ["DeviceName", "Concentration(mML)", "Annealing_T(C)", "Molecular_Weight", "Heavy_Atom_Count", "Rigid_Bond_Count", "Rotatable_Bond_Count", "Topological_Polar_Surface_Area", "Complexity", "Hydrogen_Bond_Donor_Count", "Hydrogen_Bond_Acceptor_Count", "H_Count", "C_Count", "Cl_Count", "Br_Count", "I_Count", "Halide_fraction", "Heteroatom_carbon_ratio"]
# SignificativeNEW (no TPSA & HBDC & HBAC)
# ProcessingData_cols = ["DeviceName", "Concentration(mML)", "Annealing_T(C)", "Molecular_Weight", "Rigid_Bond_Count", "Rotatable_Bond_Count", "Complexity", "H_Count", "C_Count", "Cl_Count", "Br_Count", "I_Count", "Halide_fraction", "Heteroatom_carbon_ratio"]
# SignificativeNEW (no TPSA & HBDC & HBAC & H_count & C_count)
# ProcessingData_cols = ["DeviceName", "Concentration(mML)", "Annealing_T(C)", "Molecular_Weight", "Rigid_Bond_Count", "Rotatable_Bond_Count", "Complexity", "Cl_Count", "Br_Count", "I_Count", "Halide_fraction", "Heteroatom_carbon_ratio"]
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

#Pearson Correlation per valutare influenza tra le variabili di processazione e le proprietà molecolari
cor = X.corr()
mask = np.zeros_like(cor)
plt.rcParams['font.family'] = "Arial"
mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(12, 12))  # Imposta la dimensione della figura a 12x12
    ax = sns.heatmap(cor, annot=True, annot_kws={"size": 10}, fmt=".2f",
                      cmap=plt.cm.RdBu_r, linewidths=0.5, vmin=-1, vmax=1, mask=mask, square=True)

    # Riduci la dimensione delle etichette dell'asse x e dell'asse y
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)

plt.show()

print("Pearson Correlation Values calcolati...")
print()
print("######################################################################")

########################################################################################################################################################################################################

# Prepariamo training e data set con una percentuale a scelta
from sklearn.model_selection import train_test_split

#Suddivisione randomica in train e test set
X_train, X_test, Y_Voc_train, Y_Voc_test = train_test_split(X, Y_Voc, test_size=0.2, random_state=42) # Test set is 20% of the total data, and the random state is set to ensure identical results for each run
print("Train e data sets suddivisi casualmente pronti...")
print()

# # Visto che voglio suddividerli in base al DeviceName, perché non svolgere prima la suddivisione sulla tabella completa Y per poi ottenere la tabella con le processing data e i valori di Voc? 
# # Seleziona la colonna "DeviceName" dalla tabella "Full_Data"
# device_names = Full_Data["DeviceName"]

# # # Stampa delle righe in cui compare la stringa nella colonna 'Nome'
# # # stringa_cercata = 'MEPEAI'
# # # risultato = Full_Data[Full_Data["DeviceName"].str.contains(stringa_cercata)]
# # # print(risultato)

# # Suddivisione tenendo conto di Concentration e annealing 
# device_initials = [ 'MEPEAI-5mML-20T', 
#                     'MEPEAI-5mML-100T', 
#                     'MEPEAI-10mML-20T', 
#                     'MEPEAI-10mML-100T', 
#                     'MEPEAI-15mML-20T', 
#                     'MEPEAI-15mML-100T', 
#                     'MEPEABr-5mML-20T', 
#                     'MEPEABr-5mML-100T', 
#                     'MEPEABr-10mML-20T', 
#                     'MEPEABr-10mML-100T', 
#                     'MEPEABr-15mML-20T', 
#                     'MEPEABr-15mML-100T', 
#                     'MEPEACl-5mML-20T', 
#                     'MEPEACl-5mML-100T', 
#                     'MEPEACl-10mML-20T', 
#                     'MEPEACl-10mML-100T', 
#                     'MEPEACl-15mML-20T', 
#                     'MEPEACl-15mML-100T', 
#                     'n-BAI-5mML-20T', 
#                     'n-BAI-5mML-100T', 
#                     'n-BAI-10mML-20T', 
#                     'n-BAI-10mML-100T', 
#                     'n-BAI-15mML-20T', 
#                     'n-BAI-15mML-100T', 
#                     'iso-BAI-5mML-20T', 
#                     'iso-BAI-5mML-100T', 
#                     'iso-BAI-10mML-20T', 
#                     'iso-BAI-10mML-100T', 
#                     'iso-BAI-15mML-20T', 
#                     'iso-BAI-15mML-100T', 
#                     'n-OAI-5mML-20T', 
#                     'n-OAI-5mML-100T', 
#                     'n-OAI-10mML-20T', 
#                     'n-OAI-10mML-100T', 
#                     'n-OAI-15mML-20T', 
#                     'n-OAI-15mML-100T',      
#                     'BBr-5mML-20T', 
#                     'BBr-5mML-100T', 
#                     'BBr-10mML-20T', 
#                     'BBr-10mML-100T', 
#                     'BBr-15mML-20T', 
#                     'BBr-15mML-100T', 
#                     'HBr-5mML-20T', 
#                     'HBr-5mML-100T', 
#                     'HBr-10mML-20T', 
#                     'HBr-10mML-100T', 
#                     'HBr-15mML-20T', 
#                     'HBr-15mML-100T',
#                     'OATsO-5mML-20T', 
#                     'OATsO-5mML-100T', 
#                     'OATsO-10mML-20T', 
#                     'OATsO-10mML-100T', 
#                     'OATsO-15mML-20T', 
#                     'OATsO-15mML-100T'
#                     ]

# # Only 20T
# device_initials = [ 'MEPEAI-5mML-20T',  
#                     'MEPEAI-10mML-20T',  
#                     'MEPEAI-15mML-20T',  
#                     'MEPEABr-5mML-20T', 
#                     'MEPEABr-10mML-20T',  
#                     'MEPEABr-15mML-20T',  
#                     'MEPEACl-5mML-20T',  
#                     'MEPEACl-10mML-20T',  
#                     'MEPEACl-15mML-20T',  
#                     'n-BAI-5mML-20T',  
#                     'n-BAI-10mML-20T',  
#                     'n-BAI-15mML-20T',  
#                     'iso-BAI-5mML-20T', 
#                     'iso-BAI-10mML-20T', 
#                     'iso-BAI-15mML-20T', 
#                     'n-OAI-5mML-20T', 
#                     'n-OAI-10mML-20T', 
#                     'n-OAI-15mML-20T',   
#                     'BBr-5mML-20T', 
#                     'BBr-10mML-20T', 
#                     'BBr-15mML-20T', 
#                     'HBr-5mML-20T', 
#                     'HBr-10mML-20T',  
#                     'HBr-15mML-20T', 
#                     'OATsO-5mML-20T',  
#                     'OATsO-10mML-20T',  
#                     'OATsO-15mML-20T' 
#                     ]

# # Only 20T e 10mml
# device_initials = ['MEPEAI-10mML-20T',     
#                     'MEPEABr-10mML-20T',      
#                     'MEPEACl-10mML-20T',     
#                     'n-BAI-10mML-20T',  
#                     'iso-BAI-10mML-20T', 
#                     'n-OAI-10mML-20T', 
#                     'BBr-10mML-20T', 
#                     'HBr-10mML-20T',  
#                     'OATsO-10mML-20T',  
#                     ]

# # Only 20T e 15mml
# device_initials = [ 'MEPEAI-15mML-20T',    
#                     'MEPEABr-15mML-20T',  
#                     'MEPEACl-15mML-20T',  
#                     'n-BAI-15mML-20T',  
#                     'iso-BAI-15mML-20T',  
#                     'n-OAI-15mML-20T',   
#                     'BBr-15mML-20T', 
#                     'HBr-15mML-20T', 
#                     'OATsO-15mML-20T' 
#                     ]

# # Trova gli indici per ciascuna sigla di dispositivo nella colonna "DeviceName"
# device_indices = [np.where(device_names.str.startswith(initial))[0] for initial in device_initials]

# # Dividi gli indici dei dati per ciascuna sigla in train e test
# train_indices, test_indices = [], []
# for indices in device_indices:
#     train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
#     train_indices.extend(train_idx)
#     test_indices.extend(test_idx)

# # Converti gli indici in array numpy
# train_indices = np.array(train_indices)
# test_indices = np.array(test_indices)

# # Utilizzo gli indici per ottenere i Full Data di train e test
# Full_train = Full_Data.iloc[train_indices]
# Full_test = Full_Data.iloc[test_indices]

# # print(Full_Data, Full_train, Full_test)
# # print("Full_Data:", Full_Data.shape,"Full_train:", Full_train.shape,"; Full_test:", Full_test.shape)

# # Infine li suddivido in dati di processazione e variabile target -> Voc
# X_train = Full_train[ProcessingData_cols]
# Y_Voc_train = Full_train["Voc(mV)"]
# X_test = Full_test[ProcessingData_cols]
# Y_Voc_test = Full_test["Voc(mV)"]

# print(X_train, Y_Voc_train, X_test, Y_Voc_test)

# print("X_train:", X_train.shape,"; Y_Voc_train:", Y_Voc_train.shape)
# print("X_test:", X_test.shape,"; Y_Voc_test:", Y_Voc_test.shape)

# # Conteggio per verificare suddivisione 
# for initial in device_initials:
#     train_count = (X_train["DeviceName"].str.startswith(initial)).sum()
#     test_count = (X_test["DeviceName"].str.startswith(initial)).sum()
#     ratio = round(test_count / train_count, 2)
#     print(test_count, train_count, ratio, initial)
# print()    
    
# print("Train e data sets suddivisi in base ai nomi dei device pronti...")
# print()
# print("######################################################################")
# print()

########################################################################################################################################################################################################

# # Plotting the distribution of molecular weight vs. concentration
# fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=False)
# plt.style.use('default')
# plt.tight_layout()
# sns.distplot(X["Halide_fraction"],ax=axes[0])
# sns.distplot(X["Annealing_T(C)"],ax=axes[1])
# # Imposta lo spazio orizzontale tra i subplots
# plt.subplots_adjust(wspace=0.2)

########################################################################################################################################################################################################

#Applichiamo una Standardization così da eliminare il bias legato ad una particolare variabile a causa della sua grandezza numerica.
from sklearn.preprocessing import StandardScaler, RobustScaler

# # Decommentare se non si vuole svolgere la standardizzazione
# X_noname = X_stand = X.drop("DeviceName", axis=1)
# X_train_noname = X_train.drop("DeviceName", axis=1)
# X_train_stand_no = X_train.drop("DeviceName", axis=1)
# X_test_stand = X_test.drop("DeviceName", axis=1)

#Standardizzazione valori di input con Standard Scaler
scaler = StandardScaler()
X_noname = X.drop("DeviceName", axis=1)
X_train_noname = X_train.drop("DeviceName", axis=1)
X_test_noname = X_test.drop("DeviceName", axis=1)
scaler.fit(X_noname)

X_stand = scaler.transform(X_noname)
X_train_stand_no = X_train_stand = scaler.transform(X_train_noname)
X_test_stand = scaler.transform(X_test_noname) 

print("Standardizzazione dei valori di input svolta...")
print()
print("######################################################################")
print()

#Standardizzazione valori di ouput con Standard Scaler
scaler.fit(Y_Voc)

Y_Voc = scaler.transform(Y_Voc)
Y_Voc_train = scaler.transform(Y_Voc_train)
Y_Voc_test_stand = scaler.transform() 

print("Standardizzazione dei valori di output svolta...")
print()
print("######################################################################")
print()

#Shuffle dei dati 
np.random.seed(42)
i_rand = np.arange(X_train_stand_no.shape[0])
np.random.shuffle(i_rand)
X_train_stand = np.array(X_train_stand_no)[i_rand]
Y_Voc_train = np.array(Y_Voc_train)[i_rand]

print("Shuffle dei dati svolto...")
print()
print("######################################################################")
print()

# #Standardizzazione valori di input con Robust Scaler per diminuire influenza degli outliners
# scaler = RobustScaler()

# X_noname = X.drop("DeviceName", axis=1)
# X_train_noname = X_train.drop("DeviceName", axis=1)
# X_test_noname = X_test.drop("DeviceName", axis=1)
# scaler.fit(X_noname)

# X_stand = scaler.transform(X_noname)
# X_train_stand_no = X_train_stand = scaler.transform(X_train_noname)
# X_test_stand = scaler.transform(X_test_noname) 

# print("Standardizzazione robusta dei valori di input svolta...")
# print()

#Visualize the mean and variance prior and after standardization
# fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
# plt.style.use('default')
# ax[0].set_title("Mean")
# ax[0].scatter(np.arange(X_train_noname.shape[1]), np.mean(X_train_noname, axis=0), s=10, label='X_train', c="red")
# ax[0].scatter(np.arange(X_train_stand_no.shape[1]), np.mean(X_train_stand_no, axis=0), s=10, label='X_train_stand', c="blue")
# ax[0].legend()

# ax[1].set_title("Variance")
# ax[1].scatter(np.arange(X_train_noname.shape[1]), np.var(X_train_noname, axis=0), s=10, label='X_train', c="red")
# ax[1].scatter(np.arange(X_train_stand_no.shape[1]), np.var(X_train_stand_no, axis=0), s=10, label='X_train_stand', c="blue")
# ax[1].set_yscale('log')
# ax[1].legend()
# plt.show()

########################################################################################################################################################################################################

#Visulization functions
def prediction_vs_ground_truth_fig(y_train, y_train_hat, y_test, y_test_hat, model_name):
    from sklearn import metrics
    fontsize = 12
    plt.figure(figsize=(6,5))
    plt.style.use('default')
    plt.rc('xtick', labelsize=fontsize)
    plt.rc('ytick', labelsize=fontsize)
    plt.rcParams['font.family']="Arial"
    a = plt.scatter(y_train, y_train_hat, s=25,c='#b2df8a')
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k:', lw=1.5)
    plt.xlabel('Observation', fontsize=fontsize)
    plt.ylabel('Prediction', fontsize=fontsize)
    plt.xticks([900, 950, 1000, 1050, 1100, 1150, 1200])
    plt.yticks([1050, 1100, 1150])
    plt.tick_params(direction='in')
    #plt.text(450,80,'Scaled',family="Arial",fontsize=fontsize)
    plt.xlim([1000,1200]) 
    plt.ylim([1000,1200])
    plt.title('{} - Train RMSE: {:.2e}, Test RMSE: {:.2e}'.format(model_name, np.sqrt(metrics.mean_squared_error(y_train, y_train_hat)), np.sqrt(metrics.mean_squared_error(y_test, y_test_hat))), fontsize=fontsize)
    b = plt.scatter(y_test, y_test_hat, s=25,c='#1f78b4')
    plt.legend((a,b),('Train','Test'),fontsize=fontsize,handletextpad=0.1,borderpad=0.1)
    plt.rcParams['font.family']="Arial"
    plt.tight_layout()
    #plt.savefig('Name.png', dpi = 1200)
    plt.show()

########################################################################################################################################################################################################

# Linear Regression
from sklearn.linear_model import LinearRegression

lr_regressor = LinearRegression()

# Fit to the training set
lr = lr_regressor.fit(X_train_stand, Y_Voc_train)

# Perform predictions on both training and test sets
Y_Voc_train_LR = lr_regressor.predict(X_train_stand)
Y_Voc_test_LR = lr_regressor.predict(X_test_stand)

print("Linear regressor trainato...")
print()
print("######################################################################")
print()

# Visualize the results
prediction_vs_ground_truth_fig(Y_Voc_train, Y_Voc_train_LR, Y_Voc_test, Y_Voc_test_LR, "Linear Regression")
# print(lr_regressor.coef_)

# importances = lr.coef_
# features = (list(X_noname.columns))
# color_list = np.where(importances>0,'#ca0020','#0571b0')

# ax = plt.barh(features,abs(importances), color = color_list)
# plt.rcParams['font.sans-serif'] = "Arial"
# plt.rcParams.update({'font.size': 12})
# plt.xlabel('Linear regressor coefficients')

########################################################################################################################################################################################################

# K Nearest Neighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
    
def rfr_model(X, y):
    # Perform Grid-Search
    gsc = GridSearchCV(
        estimator=KNeighborsRegressor(),
        param_grid={'weights': ['uniform', 'distance'],
                    'n_neighbors': range(2,20),
                    'algorithm': ['ball_tree','kd_tree','brute']
                    },
        cv=5, 
        scoring= 'neg_mean_squared_error',
        verbose=0,
        n_jobs=-1)
    
    grid_result = gsc.fit(X, y)
    best_params = grid_result.best_params_
    knn_regressor = KNeighborsRegressor(**best_params)  # Utilizza i migliori parametri trovati
    best_score = grid_result.best_score_
    
    return knn_regressor, best_params, best_score

knn_regressor, best_params, best_score = rfr_model(X_train_stand, Y_Voc_train)
print('The best knn parameters: ', best_params, "score: ", best_score)

# knn_regressor = KNeighborsRegressor(algorithm='kd_tree',
#                                     n_neighbors=5,
#                                     weights='uniform')

#Fit to the training set
knn_regressor.fit(X_train_stand, Y_Voc_train)
# Perform predictions on both training and test sets
Y_Voc_train_KN = knn_regressor.predict(X_train_stand)
Y_Voc_test_KN = knn_regressor.predict(X_test_stand)

print("K Nearest Neighbors trainato...")
print()
print("######################################################################")
print()

# Visualize the results
prediction_vs_ground_truth_fig(Y_Voc_train, Y_Voc_train_KN, Y_Voc_test, Y_Voc_test_KN, "K Nearest Neighbors")

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
#                                      n_estimators=100,
#                                      random_state=42)

# Fit to the training set
rf_regressor.fit(X_train_stand, Y_Voc_train)
# Perform predictions on both training and test sets
Y_Voc_train_RF = rf_regressor.predict(X_train_stand)
Y_Voc_test_RF = rf_regressor.predict(X_test_stand)

print("Random Forest trainato...")
print()
print("######################################################################")
print()

#Visualize the results
prediction_vs_ground_truth_fig(Y_Voc_train, Y_Voc_train_RF, Y_Voc_test, Y_Voc_test_RF, "Random Forest")
# rf_regressor.feature_importances_

########################################################################################################################################################################################################

# Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

def rfr_model(X, y):
# Perform Grid-Search
    gsc = GridSearchCV(
        estimator=GradientBoostingRegressor(random_state=42),
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
    gb_regressor = GradientBoostingRegressor(**best_params)  # Utilizza i migliori parametri trovati
                
    return gb_regressor, best_params

gb_regressor, best_params = rfr_model(X_train_stand, Y_Voc_train)
print('The best gb parameters: ', best_params)

# gb_regressor = GradientBoostingRegressor(max_depth = 2,
#                                          n_estimators=10, 
#                                          random_state=42)

# Fit to the training set
gb_regressor.fit(X_train_stand, Y_Voc_train)
# Perform predictions on both training and test sets
Y_Voc_train_gb = gb_regressor.predict(X_train_stand)
Y_Voc_test_gb = gb_regressor.predict(X_test_stand)

print("Gradient Boosting trainato...")
print()
print("######################################################################")
print()

# Visualize the results
prediction_vs_ground_truth_fig(Y_Voc_train, Y_Voc_train_gb, Y_Voc_test, Y_Voc_test_gb, "Gradient Boosting")

########################################################################################################################################################################################################

# SVR
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

def rfr_model(X, y):
# Perform Grid-Search
    gsc = GridSearchCV(
        estimator=SVR(),
        param_grid={
            "C": [1e1,50,1e2,5e2,1e3],
            "kernel": ['linear', 'poly', 'rbf', 'sigmoid']
            #"gamma": 'auto'
        },
        cv=5, 
        scoring= 'neg_mean_squared_error', 
        verbose=0,
        n_jobs=-1)
    
    grid_result = gsc.fit(X, y)
    best_params = grid_result.best_params_
    svr_regressor = SVR(**best_params)  # Utilizza i migliori parametri trovati
    
    return svr_regressor, best_params

svr_regressor, best_params = rfr_model(X_train_stand, Y_Voc_train)
print('The best svr parameters: ', best_params)

# svr_regressor = SVR(C=10,kernel='rbf')

# Fit to the training set
svr_regressor.fit(X_train_stand, Y_Voc_train)
# Perform predictions on both training and test sets
Y_Voc_train_svr = svr_regressor.predict(X_train_stand)
Y_Voc_test_svr = svr_regressor.predict(X_test_stand)

print("SVR trainato...")
print()
print("######################################################################")
print()

# Visualize the results
prediction_vs_ground_truth_fig(Y_Voc_train, Y_Voc_train_svr, Y_Voc_test, Y_Voc_test_svr, "SVR")

########################################################################################################################################################################################################

#NB. sotto consiglio di Ian utilizzo 'activation': ['logistic'] e 'solver': ['adam'], dovrebbero essere i migliori per il lavoro che stiamo svolgendo.
# Adam funziona solo con relu come activation.

# Neural Network
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

def rfr_model(X, y):
# Perform Grid-Search
    gsc = GridSearchCV(
        estimator=MLPRegressor(max_iter=1000),
        param_grid = {
            'hidden_layer_sizes': [(128, 256, 64,), (32,64,128,64,32,16,8)],
            'activation': ['relu'],
            'solver': ['adam'],
            'alpha': [0.0001,0.001,0.01, 0.05],
            'learning_rate': ['adaptive', 'constant'],
            'learning_rate_init': [0.001,0.01,0.1]
        },
        cv=5, 
        scoring= 'neg_mean_squared_error', 
        verbose=0,
        n_jobs=-1)
    
    grid_result = gsc.fit(X, y)
    best_params = grid_result.best_params_
    nn_regressor = MLPRegressor(**best_params)  # Utilizza i migliori parametri trovati
    
    return nn_regressor, best_params

# nn_regressor, best_params = rfr_model(X_train_stand, Y_Voc_train)
# print('The best nn parameters: ', best_params)

# nn_regressor = MLPRegressor(hidden_layer_sizes =(128,256,64), max_iter=100,
#                             solver='sgd',learning_rate='adaptive',#momentum=0.9,
#                             alpha=0.1,activation='logistic',random_state=42)

nn_regressor = MLPRegressor(hidden_layer_sizes =(32,64,128,64,32,16,8),max_iter=1000,
                            solver='adam',alpha=0.0001, activation='relu',random_state=42,
                            learning_rate_init=0.001)

# nn_regressor = MLPRegressor(hidden_layer_sizes =(32,64,128,64,32,16,8),max_iter=100,
#                             solver='lbfgs',alpha=0.01,activation='logistic',random_state=42)

#NB: quello con sgd ottiene risultati pessimi su train e test, quello con adam funziona bene (converge)
#    ma generalizza malissimo, quello con lbfgs funziona bene (converge e non posso visualizzare 
#    la loss function) e generalizza bene.

# Fit to the training set
nn_regressor.fit(X_train_stand, Y_Voc_train)
# Perform predictions on both training and test sets
Y_Voc_train_MLP = nn_regressor.predict(X_train_stand)
Y_Voc_test_MLP = nn_regressor.predict(X_test_stand)

print("Neural Network (MLP) trainato...")
print()
print("######################################################################")
print()

#Visualize the results
prediction_vs_ground_truth_fig(Y_Voc_train, Y_Voc_train_MLP, Y_Voc_test, Y_Voc_test_MLP, "Neural Network")

#Visualizzazione convergenza della loss function del MLP con diversi metodi -> solo per "adam" e "sgd"
# loss_values = nn_regressor.loss_curve_
# plt.plot(loss_values)
# plt.xlabel('Iterazione')
# plt.ylabel('Loss')
# plt.title('Funzione di Loss durante il training')
# plt.yscale('log')  # Imposta l'asse y in scala logaritmica
# plt.show()

########################################################################################################################################################################################################

#Models Validation with k-fold

print("Inizio validazione modelli...")
print()

from sklearn.model_selection import cross_validate, cross_val_score

#estraggo minimo e massimo dei valori di Voc per avere un valore in percentuale del RMSE

max_Voc = max(Y_Voc)
min_Voc = min(Y_Voc)
range_Voc = max_Voc - min_Voc

n_cv = 3

# Obtain cross-validation scores of various regressors
lr_d = cross_validate(lr_regressor, X_train_stand, Y_Voc_train, cv = n_cv, scoring='neg_mean_squared_error', return_estimator = True, return_train_score = True)
knn_d = cross_validate(knn_regressor, X_train_stand, Y_Voc_train, cv = n_cv, scoring='neg_mean_squared_error', return_estimator = True, return_train_score = True)
rf_d = cross_validate(rf_regressor, X_train_stand, Y_Voc_train, cv = n_cv, scoring='neg_mean_squared_error', return_estimator = True, return_train_score = True)
gb_d = cross_validate(gb_regressor, X_train_stand, Y_Voc_train, cv = n_cv, scoring='neg_mean_squared_error', return_estimator = True, return_train_score = True)
svr_d = cross_validate(svr_regressor, X_train_stand, Y_Voc_train, cv = n_cv, scoring='neg_mean_squared_error', return_estimator = True, return_train_score = True)
nn_d = cross_validate(nn_regressor, X_train_stand, Y_Voc_train, cv = n_cv, scoring='neg_mean_squared_error', return_estimator = True, return_train_score = True)

lr_score = lr_d["test_score"]
knn_score = knn_d["test_score"]
rf_score = rf_d["test_score"]
gb_score = gb_d["test_score"]
svr_score = svr_d["test_score"]
nn_score = nn_d["test_score"]

# Plot the cross-validation scores of the various regressors
model = ['Linear Regression', 'K Nearest Neighbors', 'Random Forest', 'Gradient Boosting', 'SVR', 'Neural Network']
scores = [np.mean(np.sqrt(-lr_score)), np.mean(np.sqrt(-knn_score)), np.mean(np.sqrt(-rf_score)), np.mean(np.sqrt(-gb_score)), np.mean(np.sqrt(-svr_score)), np.mean(np.sqrt(-nn_score))]
scores_normalized = [score / range_Voc * 100 for score in scores]
std  = [np.std(np.sqrt(-lr_score)), np.std(np.sqrt(-knn_score)), np.std(np.sqrt(-rf_score)), np.std(np.sqrt(-gb_score)), np.std(np.sqrt(-svr_score)), np.std(np.sqrt(-nn_score))]
std_normalized = [value / range_Voc * 100 for value in std]

# Visualizza valori 
fig,ax = plt.subplots(figsize=(8,6))
print('RMSE scores for LR, KNN, RF, GB, SVR, NN,: ' ,scores)
print('Std. scores for LR, KNN, RF, GB, SVR, NN: ',std)
fontsize = 12
plt.rc('xtick', labelsize=fontsize)
plt.rc('ytick', labelsize=fontsize)
ax.bar(model, scores, yerr=std, alpha=0.7, capsize=4)
ax.set_ylabel('RMSE (mV)',fontsize=fontsize)
ax.set_title('Cross Validation Scores',fontsize=fontsize)
plt.xticks(rotation=90)
plt.ylim(0,60)
plt.show()
plt.rcParams['font.family']="Arial"

print()
print(scores)
print()
print("######################################################################")
print()

# # Visualizza valori normalizzati
# fig,ax = plt.subplots(figsize=(8,6))
# print('RMSE normalized scores for LR, KNN, RF, GB, SVR, NN,: ' ,scores_normalized)
# print('Std. normalized scores for LR, KNN, RF, GB, SVR, NN,: ',std_normalized)
# fontsize = 12
# plt.rc('xtick', labelsize=fontsize)
# plt.rc('ytick', labelsize=fontsize)
# ax.bar(model,scores_normalized,yerr=std_normalized,alpha=0.7, capsize=2)
# ax.set_ylabel('RMSE (%)',fontsize=fontsize)
# ax.set_title('Cross Validation Scores (%), Numero di fold: ' + str(n_cv), fontsize=fontsize)
# plt.xticks(rotation=90)
# plt.ylim(0,30)
# plt.show()
# plt.rcParams['font.family']="Arial"
# print()
# print(scores_normalized)
# print()
# print("######################################################################")
# print()

########################################################################################################################################################################################################

# Using SHAP for Interpretability 
# N.B Problema grave della libreria shap a causa del cambiamento nelle ultime versioni di numpy riguardo bool e np.bool -> risolto installando numpy version 1.23.0

import shap

print("Inizio calcolo SHAP Values...")

# Random Forest fit on train set SHAP  ################################################################################################################################################################################################

# Use shap to explain Random Forest results
explainerRF = shap.TreeExplainer(rf_regressor, check_additivity=False)

# Get SHAP values on standardized input values
shap_values_RF = explainerRF.shap_values(X_train_stand_no)

plt.figure()
shap.summary_plot(shap_values_RF, X_train_noname, plot_type = "dot", plot_size=(8, 6),
                            color=plt.get_cmap('plasma'),
                            show = False)
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams.update({'font.size': 60})

plt.title("Random Forest SHAP values", fontsize=20)

# Change the colormap of the artists, UNCOMMENT FOR DEFAULT COLORMAP
my_cmap = plt.get_cmap('viridis')
for fc in plt.gcf().get_children():
    for fcc in fc.get_children():
        if hasattr(fcc, "set_cmap"):
            fcc.set_cmap(my_cmap)

plt.show()

#### dependece variable plot #####
            
# # Extract the colormap from the summary plot
# cmap = plt.get_cmap('viridis')

# # Index of the feature "Molecular_Weight" in your dataset
# feature_index = X_train_noname.columns.get_loc("Heteroatom_carbon_ratio")

# # Get the name of the chosen feature
# chosen_feature_name = X_train_noname.columns[feature_index]

# # Create a SHAP dependence plot for the chosen feature
# fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figsize as needed
# shap.dependence_plot(
#     ind=feature_index,
#     shap_values=shap_values_RF,
#     features=X_train_stand_no,
#     feature_names=X_train_noname.columns,
#     title=f"Partial Dependence for {chosen_feature_name}",
#     ax=ax,
#     show=False
# )

# # Set the colormap for the artists of the first plot
# for fcc in ax.get_children():
#     if hasattr(fcc, "set_cmap"):
#         fcc.set_cmap(cmap)

# plt.show()

# Random Forest fit on all dataset SHAP  ################################################################################################################################################################################################

# Fit RF to all dataset 
rf_regressor.fit(X_stand, Y_Voc)

# Use shap to explain Random Forest results
explainerRF = shap.TreeExplainer(rf_regressor, check_additivity=False)

# Get SHAP values on standardized input values
shap_values_RF = explainerRF.shap_values(X_stand)

plt.figure()
shap.summary_plot(shap_values_RF, X_noname, plot_type = "dot", plot_size=(8, 6),
                            color=plt.get_cmap('plasma'),
                            show = False)
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams.update({'font.size': 60})

plt.title("Random Forest SHAP values (All data)", fontsize=20)

# Change the colormap of the artists, UNCOMMENT FOR DEFAULT COLORMAP
my_cmap = plt.get_cmap('viridis')
for fc in plt.gcf().get_children():
    for fcc in fc.get_children():
        if hasattr(fcc, "set_cmap"):
            fcc.set_cmap(my_cmap)

plt.show()

#### dependece variable plot #####
            
# # Extract the colormap from the summary plot
# cmap = plt.get_cmap('viridis')

# # Index of the feature "Molecular_Weight" in your dataset
# feature_index = X_train_noname.columns.get_loc("Heteroatom_carbon_ratio")

# # Get the name of the chosen feature
# chosen_feature_name = X_train_noname.columns[feature_index]

# # Create a SHAP dependence plot for the chosen feature
# fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figsize as needed
# shap.dependence_plot(
#     ind=feature_index,
#     shap_values=shap_values_RF,
#     features=X_train_stand_no,
#     feature_names=X_train_noname.columns,
#     title=f"Partial Dependence for {chosen_feature_name}",
#     ax=ax,
#     show=False
# )

# # Set the colormap for the artists of the first plot
# for fcc in ax.get_children():
#     if hasattr(fcc, "set_cmap"):
#         fcc.set_cmap(cmap)

# plt.show()

# Gradient Boosting SHAP ################################################################################################################################################################################################

# # Use shap to explain Gradient Boosting results
# explainerGB = shap.TreeExplainer(gb_regressor, check_additivity=False)

# # Get standardized X Dataframe
# shap_values_GB = explainerGB.shap_values(X_train_stand_no)

# plt.figure()
# shap.summary_plot(shap_values_GB, X_train_noname, plot_type = "dot", plot_size=(12, 8),
#                             color=plt.get_cmap("plasma"),
#                             show = False)
# plt.rcParams['font.sans-serif'] = "Arial"
# plt.rcParams.update({'font.size': 60})

# plt.title("Gradient Boosting SHAP values", fontsize=20)

# # Change the colormap of the artists, UNCOMMENT FOR DEFAULT COLORMAP
# my_cmap = plt.get_cmap('viridis')
# for fc in plt.gcf().get_children():
#     for fcc in fc.get_children():
#         if hasattr(fcc, "set_cmap"):
#             fcc.set_cmap(my_cmap)

# plt.tight_layout()
# print()

# SVR SHAP ################################################################################################################################################################################################

# # Use shap to explain Support Vectors model results

# import shap
# import matplotlib.pyplot as plt

# # Costruisci l'oggetto KernelExplainer
# svm_explainer = shap.KernelExplainer(svr_regressor.predict, X_train_stand_no)

# # Calcola gli SHAP values
# svm_shap_values = svm_explainer.shap_values(X_train_stand_no)

# # Plot degli SHAP values
# plt.figure(figsize=(12, 8))
# summary = shap.summary_plot(svm_shap_values, X_train_noname, plot_type="dot",
#                             color=plt.get_cmap("plasma"),
#                             show=False)

# plt.rcParams['font.sans-serif'] = "Arial"
# plt.rcParams.update({'font.size': 12})

# plt.title("Support Vectors SHAP values")

# # Opzionalmente, per cambiare il colormap
# cmap = plt.get_cmap('viridis')
# for ax in plt.gcf().get_axes():
#     for c in ax.get_children():
#         if isinstance(c, plt.PathCollection):
#             c.set_cmap(cmap)

# plt.tight_layout()
# plt.show()

# Multi Layer Perceptron SHAP ################################################################################################################################################################################################

# X_train_stand_no = shap.sample(X_train_stand_no, 1000)
# X_train_noname = shap.sample(X_train_noname, 1000)

# # Use shap to explain Multi Layer Perceptron results
# nn_explainer = shap.KernelExplainer(nn_regressor.predict, X_train_stand_no)

# # Get standardized X Dataframe
# nn_shap_values = nn_explainer.shap_values(X_train_stand_no)

# plt.figure()
# summary = shap.summary_plot(nn_shap_values, X_train_noname, plot_type = "dot",
#                             color=plt.get_cmap("plasma"),
#                            show = False)

# plt.rcParams['font.sans-serif'] = "Arial"
# plt.rcParams.update({'font.size': 60})

# # Change the colormap of the artists, UNCOMMENT FOR DEFAULT COLORMAP
# my_cmap = plt.get_cmap('viridis')
# for fc in plt.gcf().get_children():
#     for fcc in fc.get_children():
#         if hasattr(fcc, "set_cmap"):
#             fcc.set_cmap(my_cmap)

# plt.tight_layout()

########################################################################################################################################################################################################


