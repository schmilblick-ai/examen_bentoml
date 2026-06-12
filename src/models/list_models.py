import bentoml
models_info = bentoml.models.list()
print(f"{'Modèle':<30} {'Version':>17} {'R² test':>10} {'R² test-CV':>13} {'MAE':>7} {'RMSE':>7}")
print("-" * 90)
for m in models_info:
    print(f"{m.tag.name :<30} {m.tag.version :>16}  {m.info.metadata.get('r2_test'):>10}  {m.info.metadata.get('r2_t-cv'):>11} {m.info.metadata.get('mae'):>8} {m.info.metadata.get('rmse'):>8} {m.info.metadata.get('features')}")


 # ── Choisir et charger le champion à postériori ────────────────────────

# Option A — chargement manuel par nom
champion = bentoml.sklearn.load_model("admission_ridge:latest")
if False:
    # Option B — sélection automatique sur une métrique
    best_tag = max(
        bentoml.models.list(),
        key=lambda m: m.info.metadata.get("r2_test", 0)
    ).tag
    champion = bentoml.sklearn.load_model(best_tag)

# Prêt à prédire
import pandas as pd


features = champion.info.metadata.get('features')


features1 = ["GRE Score", "TOEFL Score", "University Rating", "SOP", "LOR", "CGPA", "Research"]
print(features, features1)

sample = pd.DataFrame([[320, 110, 3, 3.5, 3.5, 8.5, 1],[320, 110, 3, 3.5, 3.5, 9.7, 1]], columns=features1)

print(champion.predict(sample))

if False:
    # ── Promouvoir un modèle en production (optionnel) ─────────────────────
    # BentoML permet de tagger un modèle "champion" vs "challenger"
    saved = bentoml.sklearn.save_model(
        "admission_ridge",
        champion,
        labels = {"stage": "production"}   # ← tag métier
    )
    # Plus tard, filtrer sur ce label
    prod_models = [m for m in bentoml.models.list()
        if m.info.labels.get("stage") == "production"]