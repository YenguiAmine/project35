# WP5 — Explicabilité SHAP

## Objectif et périmètre

WP5 explique les sorties du modèle champion enregistré dans MLflow. Le calcul porte sur la probabilité d'une classe, conserve les noms des features à l'entrée du pipeline sklearn et ne prétend pas établir de causalité.

## Implémentation

- `src/project35/explainability/shap_service.py`: sélection automatique de `TreeExplainer` pour les pipelines sklearn DecisionTree/RandomForest/ExtraTrees; Permutation SHAP reste le fallback pour Logistic Regression et les modèles non-arbre.
- `django_app/predictions/explainability.py`: charge et met en cache le CSV de référence; le changement de date de modification invalide le cache.
- `django_app/predictions/views.py`: routes locales et globales intégrées au provider MLflow et à la gestion d'erreurs REST.
- `django_app/predictions/explain_serializers.py`: contrats d'entrée et de sortie documentés dans OpenAPI/Swagger.
- `tests/explainability/`: additivité, noms de colonnes, importance globale, rejet de schéma incompatible, support numérique/catégoriel mixte et signatures PNG.

Pour un modèle arbre dans un pipeline sklearn, le service applique le `ColumnTransformer` déjà entraîné à la référence et aux observations, puis explique la probabilité demandée avec `TreeExplainer(model_output="probability", feature_perturbation="interventional")`. Les contributions des colonnes one-hot sont regroupées par feature source; l'API conserve les colonnes métier et indique `explainer_method=TreeExplainer`. Le graphique global distribue les contributions par observation et feature source. Les arbres sans pipeline compatible et les autres modèles suivent le fallback model-agnostic.

Pour Permutation SHAP, les colonnes catégorielles sont encodées en nombres uniquement dans l'espace de masquage, puis décodées avant le pipeline; l'API/scoring conserve les catégories d'origine. Dans son beeswarm, les couleurs catégorielles sont des codes locaux, non une échelle ordinale. Les deux méthodes expliquent la prédiction, sans établir de causalité.

## Configuration du jeu de référence

Configurer `PROJECT35_SHAP_BACKGROUND` vers un CSV contenant exactement les features du modèle, avec les types attendus, sans cible ni identifiant. Il doit provenir du train ou d'une population de référence approuvée; ne pas utiliser le jeu test final pour expliquer les résultats de validation. Le fichier reste hors de Git et Compose monte `data/` en lecture seule.

Les valeurs par défaut limitent l'échantillon de fond à 40 lignes et l'échantillon global à 50 lignes. Régler `SHAP_MAX_BACKGROUND_ROWS`, `SHAP_MAX_GLOBAL_ROWS` et, si nécessaire, `SHAP_MAX_EVALS` selon le budget de calcul. Le coût de permutation croît avec le nombre de features et d'évaluations, puis avec les observations globales; TreeExplainer a besoin du fond de référence pour l'estimation interventional.

Exécution locale PowerShell:

```powershell
$env:PROJECT35_SHAP_BACKGROUND = "data/processed/shap_reference.csv"
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
pipenv run python django_app/manage.py runserver 127.0.0.1:8001
```

Swagger: `http://127.0.0.1:8001/api/docs/`.

## Contrats HTTP

### Explication locale

`POST /api/explain/`

```json
{
  "instance": {"balance": 1200.5, "segment": "A"},
  "class_index": 1
}
```

Réponse: nom/version du modèle et SHA-256 du CSV de référence pour l'audit, classe expliquée, échelle (`probability`), valeur de base, probabilité prédite, résidu d'additivité, contributions SHAP par feature et `waterfall_png_base64`.

### Explication globale

`GET /api/explain/global/?class_index=1`

Réponse: nom/version du modèle et SHA-256 de la référence, classe expliquée, nombre d'observations, dictionnaire mean absolute SHAP trié décroissant, graphique bar `summary_png_base64` et beeswarm `beeswarm_png_base64`.

Les champs Base64 décodés sont des images PNG. En l'absence de modèle champion, de CSV configuré ou de schéma correspondant, l'API renvoie une erreur explicite (503 pour modèle/service indisponible, 400 pour configuration/données invalides). Aucun enregistrement brut n'est écrit dans les logs.

## Livrables et vérification

- Code d'explication locale et globale plus intégration REST.
- Documentation Swagger et ce guide d'exploitation.
- Tests unitaires SHAP sur un classifieur synthétique et tests d'API Django.
- Matrice de suivi à compléter avec l'identifiant de run, la version du modèle, le hash du CSV de référence et l'environnement d'exécution.

```powershell
pipenv install --dev
pipenv run pytest tests/explainability tests/api/test_django_scoring_api.py
pipenv run python django_app/manage.py check
```

Les tests incluent un pipeline arbre avec catégories one-hot pour vérifier TreeExplainer, le regroupement vers les features sources, l'additivité et les graphiques. La validation synthétique vérifie les mécanismes, pas la pertinence métier. Les comparatifs au 2026-11-21 montrent qu'aucun candidat arbre ne satisfait les marges F1/AUC face au champion Logistic Regression; celui-ci utilise donc Permutation SHAP, et le service choisit TreeExplainer lorsqu'un pipeline d'arbre est servi.
