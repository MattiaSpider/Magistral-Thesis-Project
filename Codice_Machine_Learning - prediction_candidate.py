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
TableCandidate = "Cationi_candidati_ML.txt"

pd.set_option('display.max_rows', 10, 'display.max_columns', 10)

# The molecular features & processing conditions are loaded as X
#All
# ProcessingData_cols = ["DeviceName", "Concentration(mML)", "Annealing_T(C)", "Molecular_Weight", "Rigid_Bond_Count", "Rotatable_Bond_Count", "Topological_Polar_Surface_Area", "Hydrogen_Bond_Donor_Count", "Hydrogen_Bond_Acceptor_Count", "Heavy_Atom_Count", "Complexity", "H_Count", "C_Count", "Cl_Count", "Br_Count", "I_Count", "Halide_fraction", "Heteroatom_carbon_ratio", "Heteroatom_count"]
#Significative
#ProcessingData_cols = ["DeviceName", "Concentration(mML)", "Annealing_T(C)", "Molecular_Weight", "Rigid_Bond_Count", "Rotatable_Bond_Count", "Topological_Polar_Surface_Area", "Complexity", "Hydrogen_Bond_Donor_Count", "Hydrogen_Bond_Acceptor_Count", "H_Count", "C_Count", "Cl_Count", "Br_Count", "I_Count", "Halide_fraction", "Heteroatom_carbon_ratio"]
#SignificativeNEW (no TPSA & HBDC)
# ProcessingData_cols = ["DeviceName", "Concentration(mML)", "Annealing_T(C)", "Molecular_Weight", "Rigid_Bond_Count", "Rotatable_Bond_Count", "Complexity", "Hydrogen_Bond_Acceptor_Count", "H_Count", "C_Count", "Cl_Count", "Br_Count", "I_Count", "Halide_fraction", "Heteroatom_carbon_ratio"]
# SignificativeNEW (no TPSA & HBDC & HBAC & H_count & C_count)
ProcessingData_cols = ["DeviceName", "Concentration(mML)", "Annealing_T(C)", "Molecular_Weight", "Rigid_Bond_Count", "Rotatable_Bond_Count", "Complexity", "Cl_Count", "Br_Count", "I_Count", "Halide_fraction", "Heteroatom_carbon_ratio"]
#No Atom_count
# ProcessingData_cols = ["DeviceName", "Concentration(mML)", "Annealing_T(C)", "Molecular_Weight", "Rigid_Bond_Count", "Rotatable_Bond_Count", "Topological_Polar_Surface_Area", "Complexity", "Hydrogen_Bond_Donor_Count", "Hydrogen_Bond_Acceptor_Count", "Halide_fraction", "Heteroatom_carbon_ratio"]

# Load the table with full data
Full_Data = pd.read_csv(TableName, delim_whitespace=True, encoding="iso-8859-1")

# # Check if the data is loaded and separated correctly
# print(Full_Data)

# Creo il dataframe con solo i valori di processazione e proprietà fisico-chimiche di interesse (input)
X = Full_Data[ProcessingData_cols]

# Creo il dataframe con solo i valori di Voc (output)
Y_Voc = Full_Data["Voc(mV)"]

# Load the table with candidates data
Candidates_Data = pd.read_csv(TableCandidate, delim_whitespace=True, encoding="iso-8859-1")

# Creo il dataframe con solo i valori di processazione e proprietà fisico-chimiche di interesse (input)
X_Candidates = Candidates_Data[ProcessingData_cols]

# # Check if the data is loaded and separated correctly
# pd.set_option('display.max_rows', 10, 'display.max_columns', 5)
# print(X, Y_Voc)
# print(Candidates_Data)

print()

print("Dataframe caricati correttamente...")

print()
print("######################################################################")
print()

########################################################################################################################################################################################################

# Imposto per il training dei modelli tutti i dati raccolti in Full_Data
X_train = X
Y_Voc_train = Y_Voc

# #No standardizzazione
# X_train_stand = X_train.drop("DeviceName", axis=1)
# X_Candidates_stand = X_Candidates.drop("DeviceName", axis=1)

#Standardizzazione valori di input con Standard Scaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_noname = X_train.drop("DeviceName", axis=1)
X_Candidates_noname = X_Candidates.drop("DeviceName", axis=1)

scaler.fit(X_train_noname)
X_train_stand = scaler.transform(X_train_noname)
X_Candidates_stand = scaler.transform(X_Candidates_noname)



print("Training set standardizzato pronto...")
print()
print("######################################################################")
print()

########################################################################################################################################################################################################

# Linear Regression
from sklearn.linear_model import LinearRegression

lr_regressor = LinearRegression()

# Fit to the training set
lr = lr_regressor.fit(X_train_stand, Y_Voc_train)

print("Linear regressor trainato...")
print()
print("######################################################################")
print()

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

# knn_regressor, best_params, best_score = rfr_model(X_train_stand, Y_Voc_train)
# print('The best knn parameters: ', best_params, "score: ", best_score)

knn_regressor = KNeighborsRegressor(algorithm='kd_tree',
                                    n_neighbors=18,
                                    weights='distance')

#Fit to the training set
knn_regressor.fit(X_train_stand, Y_Voc_train)

print("K Nearest Neighbors trainato...")
print()
print("######################################################################")
print()

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
#                                      n_estimators=50,
#                                      random_state=42)

# Fit to the training set
rf_regressor.fit(X_train_stand, Y_Voc_train)

print("Random Forest trainato...")
print()
print("######################################################################")
print()

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

# gb_regressor = GradientBoostingRegressor(max_depth = 3,
#                                          n_estimators=2000, 
#                                          random_state=42)

# Fit to the training set
gb_regressor.fit(X_train_stand, Y_Voc_train)

print("Gradient Boosting trainato...")
print()
print("######################################################################")
print()

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

# svr_regressor = SVR(C=100,kernel='rbf')

# Fit to the training set
svr_regressor.fit(X_train_stand, Y_Voc_train)

print("SVR trainato...")
print()
print("######################################################################")
print()

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

print("Neural Network (MLP) trainato...")
print()
print("######################################################################")
print()

#Visualizzazione convergenza della loss function del MLP con diversi metodi -> solo per "adam" e "sgd"

# loss_values = nn_regressor.loss_curve_
# plt.plot(loss_values)
# plt.xlabel('Iterazione')
# plt.ylabel('Loss')
# plt.title('Funzione di Loss durante il training')
# plt.yscale('log')  # Imposta l'asse y in scala logaritmica
# plt.show()

########################################################################################################################################################################################################

# Perform predictions candidates
Y_Voc_predictions_LR = lr_regressor.predict(X_Candidates_stand)
Y_Voc_predictions_KNN = knn_regressor.predict(X_Candidates_stand)
Y_Voc_predictions_RF = rf_regressor.predict(X_Candidates_stand)
Y_Voc_predictions_GB = gb_regressor.predict(X_Candidates_stand)
Y_Voc_predictions_SVR = svr_regressor.predict(X_Candidates_stand)
Y_Voc_predictions_NN = nn_regressor.predict(X_Candidates_stand)

# max_index = Y_Voc.argmax()  # Ottieni l'indice del valore massimo di Y_Voc
# max_row = Full_Data.iloc[max_index]  # Estrai la riga corrispondente all'indice massimo
# print(max_row)

print()

filtered_rows = Full_Data[Full_Data['DeviceName'].str.startswith('MEPEACl-10mML-20T')]
mean_voc = filtered_rows['Voc(mV)'].mean()
print("Best device MEPEACl-10mml-20°C mean value: " + str(mean_voc))

print("Valore previsto per PEACl-10mml-20°C =", str(Y_Voc_predictions_RF[7]))

# Estrai le iniziali dei nomi dei device + concentrazione + temperatura
Candidates_Data['Iniziali_C_T'] = Candidates_Data['DeviceName'].str.extract(r'([A-Za-z\-]+-[0-9A-Za-z]+-[0-9A-Za-z]+)').astype(str)
Candidates_Data['Iniziali_C_T'] = Candidates_Data['DeviceName'].str.replace('T$', '°C')

# Plot dei valori di Voc previsti dai modelli 
predictions = [Y_Voc_predictions_LR, Y_Voc_predictions_KNN, Y_Voc_predictions_RF, Y_Voc_predictions_GB, Y_Voc_predictions_SVR, Y_Voc_predictions_NN]  # Lista di array dei dati per i diversi modelli
models = ['Linear Regression', 'K Nearest Neighbors', 'Random Forest', 'Gradient Boosting', 'SVR', 'Neural Network']  # Aggiungi le etichette per gli altri modelli

for candidate, prediction in zip(Candidates_Data['Iniziali_C_T'], Y_Voc_predictions_RF):
    print(candidate, prediction)

for i, prediction in enumerate(predictions):
    model = models[i]  # Ottieni il nome del modello corrente
    sns.set_theme(style="ticks")
    fig, ax = plt.subplots(figsize=(6, 15))  
    
    sns.boxplot(x=prediction, y=Candidates_Data['Iniziali_C_T'], orient='h', width=0.5, palette="vlag", showmeans=True, meanline=True, meanprops={'color': 'black'})
    sns.stripplot(x=prediction, y=Candidates_Data['Iniziali_C_T'], size=10, linewidth=0.4, edgecolor='black', marker='d', palette="vlag", hue=Candidates_Data['Iniziali_C_T'])
    plt.xlabel('Voc (mV)')  # Scambia le etichette x e y
    plt.ylabel('Candidates')
    # plt.title('Voc previste su cationi candidati (' + model + ')')
    
    ax.set_yticklabels(Candidates_Data['Iniziali_C_T'], rotation=0) 
    
    plt.legend().remove()
    
    plt.show()
