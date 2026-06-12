# A propos de ce projet Bentoml et selection universitaire

## key learnings summary
As starting point, this container is empty shell

Donc le but ici va être de livrer un service ML reproductible

## Initialisation ...
On initialise uv `uv init`, on complète un pyproject.toml standard pour démarrer, puis `uv sync`
Alors pas si simple, le fichier .python-version se voit coller un 3.10, et l'upgrade à 3.12 produit des erreurs de dépendances

Je doit renoncer à la version spécifiée de pydantic, j'enlève la version dans le toml pour cet asset, voyons de quoi on hérite:
```python
Resolved 129 packages in 3ms
│   ├── pydantic v2.13.4
│   │   ├── pydantic-core v2.46.4  -> was 2.41.3 yanked
│   ├── pydantic v2.13.4 (*)
├── pydantic v2.13.4 (*)
```

Bon, on passe en 2.46, qui vivra verra !
a noté la subtilité, le source du pyproject.toml sur une autre machine est parti d'un /home/ubuntu/.venv/bin/python3, qui passe crème.
ici en local /usr/bin/python3, on est branché au départ sur le python3 du localhost, en général très difficile à la négotiation.

Bref ...

## Préparation des données
- l'accés au données
chargons les données dans le dossier `data/raw`. Pour cela, vous devez utiliser le lien suivant: https://assets-datascientest.s3.eu-west-1.amazonaws.com/MLOPS/bentoml/admission.csv

```curl https://assets-datascientest.s3.eu-west-1.amazonaws.com/MLOPS/bentoml/admission.csv -o data/raw/admission.csv``` 

ou bien on prépare un scripte d'importation python dans src/data/import_raw_data.py avec un peu plus de test
```bash
mkdir src/data
vi src/data/import_raw_data.py
```
pour une fonction de load plus générique, interactive
`uv run src/data/import_raw_data.py`


- Le nettoyage des données: éliminer les valeurs manquantes, corriger les erreurs, etc.



- La transformation des données: normalisation, standardisation, encodage des variables catégoriques, etc.
- La division des données: créer des ensembles de formation, de validation et de test.

## Création du modèle - par Validation Croisé et Tuning
- La sélection d'un algorithme de ML adapté à votre problème.
- L'entraînement du modèle sur vos données de formation.
- L'évaluation de la performance du modèle avec vos données de test.


### le choix du modèle
bon alors nous avonc une variable continue "Chance of Admit", une probabilité sur l'intervalle [0,1]
et 7 variables explicatives

https://www.kaggle.com/code/lavanyaanandm/predicting-admissions-chances#Data-Definition

GRE Score: Graduate Record Exam (GRE) score. The score will be out of 340 points (numeric)

TOEFL Score: Test of English as a Foreigner Language2 (TOEFL) score, which will be out of 120 points (numeric)

University Rating: University Rating (Uni.Rating) that indicates the Bachelor University ranking among the other universities. The score will be out of 5 (numeric)

SOP: Statement of purpose (SOP) which is a document written to show the candidate's life, ambitious and the motivations for the chosen degree/ university. The score will be out of 5 points (numeric)

LOR: Letter of Recommendation Strength (LOR) which verifies the candidate professional experience, builds credibility, boosts confidence and ensures your competency. The score is out of 5 points (numeric)

CGPA: Undergraduate GPA (CGPA) out of 10 (numeric) Cumulative Grade Point Average

Research: Research Experience that can support the application, such as publishing research papers in conferences, working as research assistant with university professor (either yes or no) (categorical)

Chance of Admit: One dependent variable can be predicted which is chance of admission, that is according to the input given will be ranging from 0 to 1 (numeric).

```
 0   GRE Score          500 non-null    int64   
 1   TOEFL Score        500 non-null    int64   
 2   University Rating  500 non-null    int64   
 3   SOP                500 non-null    float64 Categ
 4   LOR                500 non-null    float64 Categ
 5   CGPA               500 non-null    float64 Note
 6   Research           500 non-null    int64   Categ
 7   Chance_of_Admit    500 non-null    float64 Target

       GRE Score   TOEFL Score  University  SOP         LOR        CGPA        Research          Chance_of_Admit
                                Rating    
count  500.000000   500.000000  500.000000  500.000000  500.00000  500.000000  500.000000        500.00000
mean   316.472000   107.192000    3.114000    3.374000    3.48400    8.576440    0.560000          0.72174
std     11.295148     6.081868    1.143512    0.991004    0.92545    0.604813    0.496884          0.14114
min    290.000000    92.000000    1.000000    1.000000    1.00000    6.800000    0.000000          0.34000
25%    308.000000   103.000000    2.000000    2.500000    3.00000    8.127500    0.000000          0.63000
50%    317.000000   107.000000    3.000000    3.500000    3.50000    8.560000    1.000000          0.72000
75%    325.000000   112.000000    4.000000    4.000000    4.00000    9.040000    1.000000          0.82000
max    340.000000   120.000000    5.000000    5.000000    5.00000    9.920000    1.000000          0.97000
```

On va introduire différent modèle pour tester

On va prendre en compte la standardisation des modèles en fonction de chaque pipeline

En effet, les disparités de valeurs sur les variables peuvent avoir une influences sur certains modèle quand à la regression

Donc notre gradcv sera fait avec des standardScalers ajusté par modèle.

| Modèle     | Normalisation nécessaire ? | Pourquoi |
| ---------- | -------------------------- | -------- |
| Linear     |✅ Oui |  Les coefficients sont directement comparables uniquement si les features sont à la même échelle. Sans ça, le coefficient de GRE (300-340) est artificiellement petit vs CGPA (0-10) |
| Ridge      |✅ Oui, impératif |   La pénalité L2 s'applique sur les coefficients bruts ; elle pénalise injustement les features à grande échelle sans normalisation |
| Lasso      | ✅ Oui, impératif |  Même raison que Ridge ; sans normalisation, Lasso élimine des features pour de mauvaises raisons (échelle, pas importance réelle) |
| Random Forest   | ❌ Non |  Les arbres font des splits binaires sur des seuils ; l'échelle absolue n'a aucun impact |
| Gradient Boosting  | ❌ Non |   Même logique que RF ; invariant aux transformations monotones  |

### Note sur la modélisation
Nombre de combinaisons testées :: 

Random Forest a 4×5×4 = 80 combinaisons × 5 folds = 400 fits. 

Gradient Boosting a 4×4×4 = 64 × 5 = 320 fits. 

Sur un dataset de 500 lignes c'est rapide, mais n_jobs=-1 est important pour paralléliser.

refit=True :: après avoir trouvé les meilleurs paramètres par cross-validation, sklearn réentraîne automatiquement le modèle sur tout X_train. C'est ce modèle-là qu'on évalue ensuite sur X_test.

cv_r2 vs r2_test :: le cv_r2 est la métrique de sélection (moyenne sur les 5 folds du train set), le r2_test est l'évaluation finale sur des données jamais vues. Si r2_test > cv_r2, c'est normal :: le modèle final est entraîné sur plus de données. Si r2_test << cv_r2, c'est un signal de fuite de données.

```
── Résultats finaux ──────────────────────────────────────────────
Modèle                  R² CV  R² test      MAE     RMSE
────────────────────────────────────────────────────────
Linear                    -     0.8256   0.0427   0.0603
Ridge                  0.7977   0.8257   0.0429   0.0603
Lasso                  0.7988   0.8255   0.0425   0.0603
Random Forest          0.7739   0.8159   0.0429   0.0619
Gradient Boosting      0.7850   0.8078   0.0450   0.0633
```

On penserait que Ridge serait le gagnant - à comparer avec différent solver aussi
```python
param_grid = {
    'alpha': [0.1,0.5, 1.0,2.0,5.0,8.0, 10.0,15.0,50.0,75.0, 100.0],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
}
```
Donc encore un petit malin qui à trouvé que en changeant de solver sur un ridge
MSE : 0.003632118694402309
RMSE : 0.06026706143825422
R² : 0.8257313189224782

Meilleurs paramètres : {'alpha': 5.0, 'solver': 'sag'}
Meilleur score (MSE) : -0.003730664019407757



Ce que les chiffres disent vraiment :
- Linear, Ridge et Lasso sont quasi-identiques
- les différences sont dans la 4ème décimale. 
- C'est un résultat classique sur un petit dataset avec peu de features. 
- La régularisation n'apporte rien ici car il n'y a pas de surapprentissage à corriger sur 7 features seulement.

Random Forest et Gradient Boosting sont moins bons
- c'est contre-intuitif mais s'explique : ces modèles puissants ont besoin de plus de données pour exprimer leur avantage. 
- Sur ~400 lignes d'entraînement, ils surapprentissent même après tuning.

Le vrai signal : R² CV vs R² test
```
                  R² CV    R² test   Écart
Linear              -       0.826      - 
Ridge             0.798     0.826    +0.028  ← CV sous-estime
Lasso             0.799     0.826    +0.027  ← idem
Random Forest     0.774     0.816    +0.042  ← écart plus grand
Gradient Boosting 0.785     0.808    +0.023
```

L'écart positif (R² test > R² CV) signifie que le modèle final (réentraîné sur tout X_train après GridSearchCV) bénéficie de plus de données que chaque fold de CV. C'est normal et sain, pas une fuite. Mais l'écart plus grand de RF indique qu'il est plus sensible à la taille du dataset.

Alors, quel modèle choisir ?

| Critère | Gagnant | Pourquoi |
| ------- | ------- | -------- |
| R² test pur | Ridge (0.8257) | Meilleur score, même marginalement |
| Stabilité CV | Lasso (0.7988) | Meilleur score cross-validé |
| Interprétabilité | Linear / Ridge | Coefficients directs, explicables |
| Robustesse future | Ridge | Régularisation protège si nouvelles données légèrement différentes |
| Complexité | Linear | Aucun hyperparamètre, même performance |

Ma recommandation : Ridge, mais pas pour les raisons habituelles
Pas parce qu'il est meilleur en performance  ;  les 3 modèles linéaires sont statistiquement équivalents ici. Mais pour deux raisons pragmatiques :

1. Régularisation comme assurance

Si demain le dataset s'enrichit de nouvelles features potentiellement colinéaires, Ridge tient mieux que Linear pur.

2. Le vrai plafond du problème

Un R² de 0.826 signifie que vos 7 features expliquent 82.6% de la variance de la chance d'admission. Les 17.4% restants sont 
probablement des facteurs non capturés : lettres de recommandation qualitatives, résultats d'entretien, profil de l'université d'origine... Aucun modèle ne peut dépasser ce plafond informationnel avec ces features.

Ce qui confirme notre intuition de départ sur le dataset : on prédit une probabilité à partir d'autres probabilités, ce qui crée mécaniquement une relation quasi-linéaire  :  d'où la domination des modèles linéaires.

En pratique on va commencé par sauver tous les modèles collectés dans bento

## Bentoml Model Store
- sauver et charger des modèles - un petit Makefile

Inconvénient, définir et filtrer les models se fait avec des commandes linux grep et consort

```Makefile
lst_latests:
        for l in  ~/bentoml/models/*/latest; do echo $$(cat $$l); done
rm_oldest:
        bentoml models list |grep -v "Tag" | grep -v "22:32:" | awk '{print $1}' | xargs -I {oldest} bentoml models delete {oldest} --yes
du_bento:
        du -sh ~/bentoml/models/*
```

J'ai noté qu'il devait être délicat de fournir la liste des variables pour un prédict, d'ou la mise en place de controle sur la liste des variables
et l'évitement du passage des parametretres par array. le controle pydantic ne suffit pas si en dessous on passe les variables par array. Inversé des colonnes et si vite arrivé sur de longue liste.

- librairie de modèle préentrainés - historique, version
- Accés par cache (Attention qui dit cache, dit speed *et* ressources)


