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

# __ Grilles d'hyperparamètres ______________________________________________
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
        "params": { "reg__alpha": [0.01, 0.1, 1, 10, 100],
                    "reg__solver": ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
                    }
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


# __ GridSearchCV ___________________________________________________________
results = {}

for name, cfg in param_grids.items():
    print(f"→ {name} ...", end=" ")

    if cfg["params"]:
        gs = GridSearchCV(
            estimator  = cfg["model"],
            param_grid = cfg["params"],
            cv         = 5,            # 5-fold cross-validation
            scoring    = "r2",         # métrique d'optimisation
            n_jobs     = -1,           # tous les cœurs CPU
            refit      = True          # réentraîne le meilleur modèle sur tout X_train
        )
        gs.fit(X_train, y_train)
        best_model  = gs.best_estimator_
        best_params = gs.best_params_
        cv_score    = gs.best_score_
    else:
        # Linear : pas de grid, on entraîne directement
        best_model  = cfg["model"].fit(X_train, y_train)
        best_params = {}
        cv_score    = None

    y_pred = best_model.predict(X_test)

    results[name] = {
        "best_params": best_params,
        "cv_r2":       cv_score,
        "r2_test":     r2_score(y_test, y_pred),
        "mae":         mean_absolute_error(y_test, y_pred),
        "rmse":        np.sqrt(mean_squared_error(y_test, y_pred)),
        "model":       best_model,
    }
    print(f"done — best R² CV: {cv_score:.4f}" if cv_score else "done")

# __ Affichage récapitulatif ________________________________________________
print("\n__ Résultats finaux ______________________________________________")
print(f"{'Modèle':<20} {'R² CV':>8} {'R² test':>8} {'MAE':>8} {'RMSE':>8}")
print("_" * 56)
for name, r in results.items():
    cv  = f"{r['cv_r2']:.4f}" if r['cv_r2'] else "  —   "
    print(f"{name:<20} {cv:>8} {r['r2_test']:>8.4f} {r['mae']:>8.4f} {r['rmse']:>8.4f}")

print("\n__ Meilleurs hyperparamètres _____________________________________")
for name, r in results.items():
    print(f"{name}: {r['best_params']}")


if False:
    best_m = results["Random Forest"]["model"]
    best_m.predict([[320, 110, 3, 3.5, 3.5, 8.5, 1]])  # exemple de prédiction

    # Enregistrer le modèle dans le Model Store de BentoML
    model_ref = bentoml.sklearn.save_model("accidents_rf", rf_classifier)
    print(f"Modèle enregistré sous : {model_ref}")



# __ Sauvegarder les 5 modèles après GridSearchCV _______________________
features = list(X_train.columns)
for name, r in results.items():
    # Nom normalisé pour BentoML (pas d'espaces)
    model_name = name.lower().replace(" ", "_")

    saved = bentoml.sklearn.save_model(
        name   = f"admission_{model_name}",
        model  = r["model"],           # best_estimator_ du GridSearchCV
        labels = {"dataset": "admission", "type": "regression"},
        metadata = { # métriques attachées au modèle
            "r2_cv":   round(r["cv_r2"], 4) if r["cv_r2"] else "None",
            "r2_test": round(r["r2_test"], 4),
            "r2_t-cv": round(r["cv_r2"] - r["r2_test"], 4) if r["cv_r2"] else "None",
            "mae":     round(r["mae"], 4),
            "rmse":    round(r["rmse"], 4),
            "best_params": str(r["best_params"]),
            "features": features
        }
    )
    print(f"Saved: {saved.tag}")

# __ Lister tous les modèles sauvegardés ________________________________
models_info = bentoml.models.list()
for m in models_info:
    print(f"{m.tag}  |  R² test: {m.info.metadata.get('r2_test')} |  R² test 6CV: {m.info.metadata.get('r2_t-cv')}")

if False:
    # __ Choisir et charger le champion à postériori ________________________

    # Option A — chargement manuel par nom
    champion = bentoml.sklearn.load_model("admission_ridge:latest")

    # Option B — sélection automatique sur une métrique
    best_tag = max(
        bentoml.models.list(),
        key=lambda m: m.info.metadata.get("r2_test", 0)
    ).tag
    champion = bentoml.sklearn.load_model(best_tag)

    # Prêt à prédire
    champion.predict([[320, 110, 3, 3.5, 3.5, 8.5, 1]])
    
    # __ Promouvoir un modèle en production (optionnel) _____________________
    # BentoML permet de tagger un modèle "champion" vs "challenger"
    saved = bentoml.sklearn.save_model(
        "admission_ridge",
        champion,
        labels = {"stage": "production"}   # ← tag métier
    )
    # Plus tard, filtrer sur ce label
    prod_models = [m for m in bentoml.models.list()
        if m.info.labels.get("stage") == "production"]