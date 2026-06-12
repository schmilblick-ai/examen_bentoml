import os
import bentoml
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def import_dataset(file_path, **kwargs):
    return pd.read_csv(file_path, **kwargs)

input_filepath='./data/processed'

fX_train = f"{input_filepath}/X_train.csv"
f_y = f"{input_filepath}/y_train.csv"
fX_test = f"{input_filepath}/X_test.csv"
f_ytest = f"{input_filepath}/y_test.csv"
    # Import datasets
X_train_src = import_dataset(fX_train,sep=',')
y_train = import_dataset(f_y)
X_test_src = import_dataset(fX_test,sep=',')
y_test = import_dataset(f_ytest)

scaler = StandardScaler()
print(X_train_src.head())

X_train=scaler.fit_transform(X_train_src)
X_test=scaler.transform(X_test_src)

# Modèle de régression à utiliser : Ridge Regression
ridge = Ridge()

# Définir les paramètres à tester dans la grille
param_grid = {
    'alpha': [0.1,0.5, 1.0,2.0,5.0,8.0, 10.0,15.0,50.0,75.0, 100.0],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
}

# Initialiser la recherche par grille
grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')

# Lancer la recherche des meilleurs paramètres
grid_search.fit(X_train, y_train)

# Récupérer les meilleurs paramètres et les meilleurs scores
best_model  = grid_search.best_estimator_
best_params = grid_search.best_params_
best_score  = grid_search.best_score_

print(f"Meilleurs paramètres : {best_params}")
print(f"Meilleur score (MSE) : {best_score}")
# Créer le modèle Ridge avec les meilleurs paramètres
model = Ridge(alpha=best_params['alpha'], solver=best_params['solver'])

# Entraîner le modèle sur les données d'entraînement
model.fit(X_train, y_train)

# Faire des prédictions avec le modèle chargé
y_pred = model.predict(X_test)
#single=X_test.iloc[0]
single=X_test[0,:]
#print(model.predict(single.reshape(1, -1)))
print(model.predict(single.reshape(1, -1)))

# Évaluation des performances
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred,squared=False)
r2 = r2_score(y_test, y_pred)

# Afficher les métriques
print(f"MSE : {mse}")
print(f"RMSE : {rmse}")
print(f"R² : {r2}")

# Enregistrer le modèle dans le Model Store de BentoML
#model_ref = bentoml.sklearn.save_model("admission_rd", model)

#print(f"Modèle enregistré sous : {model_ref}")
