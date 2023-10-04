# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 19:54:38 2023

@author: utente
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

TableName = "1&2&5-Rev&For-Clean(Voc-1000)_ML-means.txt"
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

#liste in cui verranno inseriti gli errori medi dei modelli per ogni ciclo 
lr_errors = []
knn_errors = []
rf_errors = []
gb_errors = []
svr_errors = []
nn_errors = []

#liste in cui verranno inseriti le std medie dei modelli per ogni ciclo 
lr_stds = []
knn_stds = []
rf_stds = []
gb_stds = []
svr_stds = []
nn_stds = []

# Validazione all but one per estrarre modello che generalizza meglio
for device_initials in all_device_initials:
    
    # Estrai l'iniziale comune dalla prima stringa nella lista
    import re 
    common_initial = re.split(r'-\d', device_initials[0])[0]
        
    # Creo train set con n-1 cationi e test set con n-esimo catione
    X_train = X[~X['DeviceName'].str.startswith(tuple(device_initials))]
    X_test = X[X['DeviceName'].str.startswith(tuple(device_initials))]
    Y_Voc_train = Y_Voc[~Y_Voc['DeviceName'].str.startswith(tuple(device_initials))]
    Y_Voc_test = Y_Voc[Y_Voc['DeviceName'].str.startswith(tuple(device_initials))]

    print()
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
    
    #Shuffle dei dati 
    np.random.seed(1)
    i_rand = np.arange(X_train_stand_no.shape[0])
    np.random.shuffle(i_rand)
    X_train_stand = np.array(X_train_stand_no)[i_rand]
    Y_Voc_train = np.array(Y_Voc_train)[i_rand]

    print("Shuffle dei dati svolto...")
    print()
    print("######################################################################")
    print()
    
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
    
    #Visualize the results
    # prediction_vs_ground_truth_fig(Y_Voc_train, Y_Voc_train_LR, Y_Voc_test, Y_Voc_test_LR, "Linear Regression")
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
    
    # knn_regressor = KNeighborsRegressor(algorithm='ball_tree',
    #                                     n_neighbors=18,
    #                                     weights='distance')
    
    #Fit to the training set
    knn_regressor.fit(X_train_stand, Y_Voc_train)
    # Perform predictions on both training and test sets
    Y_Voc_train_KN = knn_regressor.predict(X_train_stand)
    Y_Voc_test_KN = knn_regressor.predict(X_test_stand)
    
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
    
    # gb_regressor = GradientBoostingRegressor(max_depth = 4,
    #                                           n_estimators=100, 
    #                                           random_state=42)
    
    # Fit to the training set
    gb_regressor.fit(X_train_stand, Y_Voc_train)
    # Perform predictions on both training and test sets
    Y_Voc_train_gb = gb_regressor.predict(X_train_stand)
    Y_Voc_test_gb = gb_regressor.predict(X_test_stand)
    
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
    
    # svr_regressor = SVR(C=1000,kernel='rbf')
    
    # Fit to the training set
    svr_regressor.fit(X_train_stand, Y_Voc_train)
    # Perform predictions on both training and test sets
    Y_Voc_train_svr = svr_regressor.predict(X_train_stand)
    Y_Voc_test_svr = svr_regressor.predict(X_test_stand)
    
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
    # Perform predictions on both training and test sets
    Y_Voc_train_MLP = nn_regressor.predict(X_train_stand)
    Y_Voc_test_MLP = nn_regressor.predict(X_test_stand)
    
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
    
    # # No standardizzazione 
    # X_test_6_stand = X_test_6.drop("DeviceName", axis=1)
        
    # Otteniamo le medie dei valori sperimentali per ogni combinazione contenuta nella lista corrente
    Y_Voc_test_n_mean = []
    for device in device_initials:
        device_data = Y_Voc[Y_Voc['DeviceName'].str.startswith(device)]
        mean_voc_value = np.mean(device_data['Voc(mV)'])
        Y_Voc_test_n_mean.append(mean_voc_value)

    # print(Y_Voc_test_n_mean)    

    # Ottieni predizioni per ogni modello sulle 6 combinazioni dell'n-esimo catione
    Y_Voc_test_LR = lr_regressor.predict(X_test_6_stand)
    Y_Voc_test_KNN = knn_regressor.predict(X_test_6_stand)
    Y_Voc_test_RF = rf_regressor.predict(X_test_6_stand)
    Y_Voc_test_GB = gb_regressor.predict(X_test_6_stand)
    Y_Voc_test_SVR = svr_regressor.predict(X_test_6_stand)
    Y_Voc_test_NN = nn_regressor.predict(X_test_6_stand)

    # Ricaviamo l'RMSE e la std
    from sklearn import metrics
    lr_score = np.sqrt(metrics.mean_squared_error(Y_Voc_test_n_mean, Y_Voc_test_LR))
    knn_score = np.sqrt(metrics.mean_squared_error(Y_Voc_test_n_mean, Y_Voc_test_KNN))
    rf_score = np.sqrt(metrics.mean_squared_error(Y_Voc_test_n_mean, Y_Voc_test_RF))
    gb_score = np.sqrt(metrics.mean_squared_error(Y_Voc_test_n_mean, Y_Voc_test_GB))
    svr_score = np.sqrt(metrics.mean_squared_error(Y_Voc_test_n_mean, Y_Voc_test_SVR))
    nn_score = np.sqrt(metrics.mean_squared_error(Y_Voc_test_n_mean, Y_Voc_test_NN))

    lr_std = np.std(Y_Voc_test_n_mean - Y_Voc_test_LR)
    knn_std = np.std(Y_Voc_test_n_mean - Y_Voc_test_KNN)
    rf_std = np.std(Y_Voc_test_n_mean - Y_Voc_test_RF)
    gb_std = np.std(Y_Voc_test_n_mean - Y_Voc_test_GB)
    svr_std = np.std(Y_Voc_test_n_mean - Y_Voc_test_SVR)
    nn_std = np.std(Y_Voc_test_n_mean - Y_Voc_test_NN)
    
    max_Voc = max(Y_Voc_test)
    min_Voc = min(Y_Voc_test)
    range_Voc = max_Voc - min_Voc
    
    scores = [lr_score, knn_score, rf_score, gb_score, svr_score, nn_score]
    stds = [lr_std, knn_std, rf_std, gb_std, svr_std, nn_std]
    
    scores_normalized = [score / range_Voc * 100 for score in scores]
    stds_normalized = [std / range_Voc * 100 for std in stds]
    
    # # Visualizza valori 
    # fig,ax = plt.subplots(figsize=(8,6))
    # print('RMSE scores for LR, KNN, RF, GB, SVR, NN,: ' ,scores)
    # print('Std. dev: ',stds)
    # fontsize = 12
    # plt.rc('xtick', labelsize=fontsize)
    # plt.rc('ytick', labelsize=fontsize)
    # model = ['Linear Regression', 'K Nearest Neighbors', 'Random Forest', 'Gradient Boosting', 'SVR', 'Neural Network']
    # ax.bar(model,scores,yerr=stds,alpha=0.7, capsize=2)
    # ax.set_ylabel('RMSE (mV)',fontsize=fontsize)
    # ax.set_title('Scores su n-esimo catione (' + common_initial + ')',fontsize=fontsize)
    # plt.xticks(rotation=90)
    # plt.ylim(0,100)
    # plt.show()
    # plt.rcParams['font.family']="Arial"
    
    # # Visualizza valori normalizzati
    # fig,ax = plt.subplots(figsize=(8,6))
    # print('RMSE scores for LR, KNN, RF, GB, SVR, NN,: ' ,scores_normalized)
    # print('Std. dev: ',stds_normalized)
    # fontsize = 12
    # plt.rc('xtick', labelsize=fontsize)
    # plt.rc('ytick', labelsize=fontsize)
    # model = ['Linear Regression', 'K Nearest Neighbors', 'Random Forest', 'Gradient Boosting', 'SVR', 'Neural Network']
    # ax.bar(model,scores_normalized,yerr=stds_normalized,alpha=0.7, capsize=2)
    # ax.set_ylabel('RMSE (%)',fontsize=fontsize)
    # ax.set_title('Scores su n-esimo catione normalizzati  (' + common_initial + ')', fontsize=fontsize)
    # plt.xticks(rotation=90)
    # plt.ylim(0,100)
    # plt.show()
    # plt.rcParams['font.family']="Arial"
    
    # Plot confronto tra dati sperimentali e dati previsi dai modelli
    data = [Y_Voc_test_n_mean, Y_Voc_test_LR, Y_Voc_test_KNN, Y_Voc_test_RF, Y_Voc_test_GB, Y_Voc_test_SVR, Y_Voc_test_NN]  # Lista di array dei dati per i diversi modelli
    
    sns.set_theme(style="ticks")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # sns.stripplot(data=data, marker='o', color='white', edgecolor='black', size=5)
    # sns.boxplot(data=data, orient='v', width=0.5, palette="vlag", showmeans=True, meanline=True, meanprops={'color': 'red'})
    sns.boxplot(data=data, orient='v', width=0.5, palette="vlag")
    sns.stripplot(data=data, size=5, linewidth=0.7, palette="vlag", edgecolor='black')
    plt.xlabel('Model')
    plt.ylabel('Voc (mV)')
    plt.title('Comparison of experimental and model-predicted data (' + common_initial + ')')
    
    # Etichette sull'asse x per i modelli
    model_labels = ['experimental data', 'Linear Regression', 'K Nearest Neighbors', 'Random Forest', 'Gradient Boosting', 'SVR', 'Neural Network']  # Aggiungi le etichette per gli altri modelli
    ax.set_xticklabels(model_labels, rotation=90)
    plt.ylim(900,1200)
    
    plt.show()
    
    #inserisco gli score ottenuti nelle liste
    lr_errors.append(lr_score)   
    knn_errors.append(knn_score)  
    rf_errors.append(rf_score)  
    gb_errors.append(gb_score)  
    svr_errors.append(svr_score)  
    nn_errors.append(nn_score)  
    
    print(rf_errors)
    
    #inserisco le std ottenute nelle liste
    lr_stds.append(lr_std)   
    knn_stds.append(knn_std)  
    rf_stds.append(rf_std)  
    gb_stds.append(gb_std)  
    svr_stds.append(svr_std)  
    nn_stds.append(nn_std)  
    
    print(rf_stds)
    
    print()
    print("All but one con", common_initial, "svolto")
    print()
    print("-------------------------------------------------------------------------")
    print()
    
#Ottengo il valore medio degli RMSE di ogni modello su tutti i cicli all but one
lr_mean_error = np.mean(lr_errors)    
knn_mean_error = np.mean(knn_errors) 
rf_mean_error = np.mean(rf_errors)   
gb_mean_error = np.mean(gb_errors)   
svr_mean_error = np.mean(svr_errors)  
nn_mean_error = np.mean(nn_errors)   

#Ottengo il valore medio delle stds di ogni modello su tutti i cicli all but one
lr_mean_std = np.mean(lr_stds)    
knn_mean_std = np.mean(knn_stds) 
rf_mean_std = np.mean(rf_stds)   
gb_mean_std = np.mean(gb_stds)   
svr_mean_std = np.mean(svr_stds)  
nn_mean_std = np.mean(nn_stds)  

ABO_scores = [lr_mean_error, knn_mean_error, rf_mean_error, gb_mean_error, svr_mean_error, nn_mean_error]
ABO_std = [lr_mean_std, knn_mean_std, rf_mean_std, gb_mean_std, svr_mean_std, nn_mean_std]

# Visualizza risultato
fig,ax = plt.subplots(figsize=(8,6))
print('All but one validation RMSE scores for LR, KNN, RF, GB, SVR, NN,: ' , ABO_scores)
print('All but one validation std scores for LR, KNN, RF, GB, SVR, NN,: ' , ABO_std)
# print('Std. dev: ',stds)
fontsize = 12
plt.rc('xtick', labelsize=fontsize)
plt.rc('ytick', labelsize=fontsize)
model = ['Linear Regression', 'K Nearest Neighbors', 'Random Forest', 'Gradient Boosting', 'SVR', 'Neural Network']
ax.bar(model, ABO_scores, yerr=ABO_std, alpha=0.7, capsize=4)
ax.set_ylabel('RMSE (mV)',fontsize=fontsize)
# ax.set_title('All but One validation scores (experimental means)',fontsize=fontsize)
plt.xticks(rotation=90)
plt.ylim(0,70)
plt.show()
plt.rcParams['font.family']="Arial"

plt.show
    
