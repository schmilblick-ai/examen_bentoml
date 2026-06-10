import sklearn
import pandas as pd 
from sklearn import ensemble
import joblib
import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
#pipeliner
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#print(joblib.__version__)

X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv')
y_test = pd.read_csv('data/processed/y_test.csv')
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)




param_grids = {
    "Linear": {
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("reg",    LinearRegression())
        ]),
        "params": {}
    },
    "Ridge": {
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("reg",    Ridge())
        ]),
        "params": { "reg__alpha": [0.01, 0.1, 1, 10, 100] }
        #            ^^^^ préfixe obligatoire avec Pipeline
    },
    "Lasso": {
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("reg",    Lasso(max_iter=5000))
        ]),
        "params": { "reg__alpha": [0.0001, 0.001, 0.01, 0.05, 0.1] }
    },
    "Random Forest": {
        "model": RandomForestRegressor(random_state=42),  # pas de scaler
        "params": {
            "n_estimators":     [50, 100, 200, 300],
            "max_depth":        [None, 3, 5, 8, 10],
            "min_samples_leaf": [1, 2, 5, 10]
        }
    },
    "Gradient Boosting": {
        "model": GradientBoostingRegressor(random_state=42),  # pas de scaler
        "params": {
            "n_estimators":  [50, 100, 200, 500],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth":     [2, 3, 4, 5]
        }
    },
}



rf_classifier = ensemble.RandomForestClassifier(n_jobs = -1)

#--Train the model
rf_classifier.fit(X_train, y_train)

#--Test the model
rf_classifier.predict(X_test)

#--Get the model accuracy
accuracy = rf_classifier.score(X_test, y_test)

print(f"Model accuracy: {accuracy}")

# test the model on a single observation
test_data = X_test.iloc[[0]]

# save test data to a txt file
test_data.to_csv('data/test_data.txt', sep = ',', index = False)
# print the actual label
print(f"Actual label: {y_test[0]}")
# print the predicted label
print(f"Predicted label: {rf_classifier.predict(test_data)[0]}")

# #--Save the trained model to a file
# model_filename = './src/models/trained_model.joblib'
# joblib.dump(rf_classifier, model_filename)
# print("Model trained and saved successfully.")


# Enregistrer le modèle dans le Model Store de BentoML
model_ref = bentoml.sklearn.save_model("accidents_rf", rf_classifier)

print(f"Modèle enregistré sous : {model_ref}")
