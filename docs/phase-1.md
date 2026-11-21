# Phase 1 — Données, contrat, EDA et baseline

## Portée

Cette phase met en place l'ingestion, la validation du contrat, le rapport qualité, les splits train/validation/test et un pipeline baseline Scikit-learn. La cible et le dataset restent volontairement non renseignés tant que le sujet officiel ne les confirme pas. Les fixtures synthétiques ne servent qu'aux tests.

## Étapes et critères de sortie

1. Copier `configs/data_contract.example.yaml` vers `configs/data_contract.yaml`, puis saisir le chemin, la cible, les colonnes ID/date et le split selon le sujet.
2. Déposer le dataset officiel sous `data/raw/` (exclu de Git s'il est volumineux ou protégé).
3. Exécuter la validation; corriger les colonnes manquantes, la cible nulle, les types et les anomalies, en conservant un rapport des corrections.
4. Examiner `duplicate_rows`, `missing_by_column`, types et distribution de la cible. Évaluer fuite de cible et variables post-événement.
5. Construire les splits reproductibles. Stratifier lorsque le problème et les effectifs le permettent; employer le split chronologique si la nature temporelle l'exige.
6. Ajuster la baseline sur le train seulement. L'imputation, le scaling et l'encodage sont dans le Pipeline afin d'être réutilisés à l'inférence.
7. Une fois le dataset officiel en place, ajouter le rapport EDA et les métriques de validation/test comme livrables de phase.

## Fichiers livrés

- `configs/data_contract.example.yaml`: configuration de référence; valeurs officielles à renseigner.
- `src/project35/data/contract.py`: chargement YAML, contrat, contrôles de schéma et rapport qualité.
- `src/project35/data/phase1.py`: splits et baseline `ColumnTransformer` + `Pipeline`.
- `scripts/phase1.py`: commande de validation/entraînement et export du rapport JSON.
- `src/project35/api/main.py`: API de développement pour tester le contrat de données, avec Swagger sur `/docs`.
- `tests/data/` et `tests/api/`: tests pytest unitaires et HTTP.
- `.github/workflows/ci.yml`: exécute pytest à chaque push et pull request.

## Commandes

```powershell
pipenv install --dev
Copy-Item configs/data_contract.example.yaml configs/data_contract.yaml
Copy-Item configs/data_provenance.example.yaml configs/data_provenance.yaml
pipenv run pytest --cov=project35 --cov-report=term-missing
pipenv run python scripts/preflight_data.py --config configs/data_contract.yaml --provenance configs/data_provenance.yaml
pipenv run python scripts/phase1.py --config configs/data_contract.yaml --provenance configs/data_provenance.yaml
pipenv run uvicorn project35.api.main:app --reload --app-dir src
```

Swagger UI: `http://127.0.0.1:8000/docs`; schéma OpenAPI: `http://127.0.0.1:8000/openapi.json`.

Docker Compose:

```powershell
docker compose up --build
pipenv run pytest --cov=project35 --cov-report=term-missing
```

## Tests attribués par rôle (à remplacer par les noms des membres)

| Rôle d'équipe | Responsabilité | Tests/livrables associés |
|---|---|---|
| Membre A — Data owner | provenance dataset, contrat, EDA, cible et qualité | `tests/data/test_phase1.py`; revue des colonnes et du rapport |
| Membre B — ML engineer | split, baseline et absence de fuite | test du split, pipeline fit/predict, catégories inconnues |
| Membre C — API engineer | FastAPI Swagger de contrat et Django REST de scoring | `tests/api/test_data_contract_api.py`, `tests/api/test_django_scoring_api.py`; contrat HTTP, Swagger et erreurs |
| Membre D — QA/MLOps | CI pytest, intégration, couverture, Docker et preuves | `tests/training/test_tracking.py`, `.github/workflows/ci.yml`, exécution Compose et rapport de démonstration |

Chaque test doit avoir un propriétaire, une assertion liée à une exigence, un résultat CI et un identifiant dans la matrice de traçabilité. Les scénarios d'API FastAPI et Django REST sont couverts dans leurs suites pytest respectives.

## Documentation par exécution

Conserver sous `reports/phase1/` le rapport JSON, la date, la version/hash du dataset, les tailles de split, les variables retenues/exclues et les anomalies connues. Ne jamais inclure de secrets ni de lignes personnelles dans les rapports ou les logs.

## WP2 — MLflow Tracking (lot suivant)

- `src/project35/training/trainer.py` entraîne sur train et n'évalue que sur validation pour le choix de la baseline.
- Il trace paramètres, F1/ROC-AUC/PR-AUC, versions, SHA-256 du CSV, signature, exemple d'entrée, rapport JSON et graphiques.
- `scripts/train.py` orchestre le run; `configs/mlflow.example.yaml` donne la configuration de tracking locale.
- `tests/training/test_tracking.py` vérifie un run complet sur un CSV synthétique temporaire, sans le confondre avec le dataset du sujet.
- Le registre et l'enregistrement de versions sont laissés au WP3; aucun alias n'est déplacé à ce stade.

Copier les configurations data/provenance et démarrer MLflow dans un premier terminal. Le dataset doit être approuvé et sa provenance/hash remplis pour permettre le fit.

```powershell
Copy-Item configs/mlflow.example.yaml configs/mlflow.yaml
pipenv run mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000
```

Après configuration du dataset officiel, lancer l'entraînement depuis un second terminal:

```powershell
pipenv run python scripts/train.py --data-config configs/data_contract.yaml --provenance configs/data_provenance.yaml --mlflow-config configs/mlflow.yaml
```

Ouvrir `http://127.0.0.1:5000` pour consulter les expériences et runs. Le backend SQLite conserve les métadonnées; les artefacts sont stockés dans `mlartifacts/`. Pour une équipe ou un déploiement partagé, remplacer SQLite par PostgreSQL et configurer un artifact store partagé.

## WP3 — Model Registry et aliases

- `src/project35/registry/service.py` enregistre le modèle `baseline_pipeline` d'un run terminé, avec des tags reliant la version au run d'origine.
- Les aliases autorisés sont `champion` et `challenger` (MLflow les affiche comme `@champion` et `@challenger`). Ils désignent une version modifiable par alias, pas une version de modèle mutée.
- Le premier alias champion peut être initialisé s'il n'existe pas. Remplacer un champion exige un `gate_report_id` dont le rapport PASS MLflow est revalidé par le service du registre (WP7); un booléen fourni par l'appelant ne suffit pas.
- `resolve_alias` renvoie la version/run/métriques; `load_alias_model` charge par URI `models:/<nom>@<alias>`.
- Le test d'intégration vérifie enregistrement, résolution, chargement du modèle et refus de déplacement de champion sans preuve de gate.

Exemple: récupérer le `run_id` imprimé par `scripts/train.py`, puis enregistrer sa première version comme champion :

```powershell
pipenv run python scripts/register_model.py --run-id <RUN_ID> --model-name project35-classifier --alias champion
```

Enregistrer un run ultérieur comme challenger :

```powershell
pipenv run python scripts/register_model.py --run-id <RUN_ID> --model-name project35-classifier --alias challenger
```

Les URI et sémantiques d'alias suivent l'API officielle MLflow Registry. Le WP4 résout l'alias de scoring et expose la version chargée dans `/health/` et les réponses API.

## WP4 — API Django REST et Swagger

- `django_app/config/` contient les paramètres et routes du service Django.
- `django_app/predictions/serializers.py` valide le lot JSON, la taille maximale et les réponses documentées.
- `django_app/predictions/services.py` résout `@champion`, charge la version MLflow puis ne recharge le modèle que si l'alias a changé. Une erreur de chargement retourne 503 sans supprimer le dernier modèle sain du cache.
- `django_app/predictions/views.py` implémente health, prédiction/probabilités et model-info; les journaux gardent request-id, version, nombre de lignes et latence, sans les features brutes.
- Swagger UI: `/api/docs/`; schéma: `/api/schema/`.
- `tests/api/test_django_scoring_api.py` couvre OpenAPI/Swagger, requête valide, schéma invalide, volume excessif, métadonnées et erreur registre; `tests/training/test_tracking.py` exerce aussi le parcours réel run → registre/alias → provider Django → prédiction.
- Le compose expose FastAPI en 8000 (contrat/développement) et Django REST en 8001 (scoring du sujet). Le service Django utilise `MLFLOW_TRACKING_URI` pour atteindre le serveur MLflow.

Commandes locales (démarrer le serveur MLflow dans un autre terminal):

```powershell
pipenv run python django_app/manage.py check
pipenv run python django_app/manage.py migrate --noinput
pipenv run python django_app/manage.py runserver 127.0.0.1:8001
pipenv run pytest tests/api/test_django_scoring_api.py
```

Swagger: `http://127.0.0.1:8001/api/docs/`.

Avec Compose:

```powershell
docker compose up --build django-api
```

Le prototype reste configuré pour accès local anonyme; une API exposée hors machine locale doit recevoir authentification et restrictions d'accès avant déploiement.

## Limitation de conformité

FastAPI/Swagger sert au contrat et aux essais de développement dès le démarrage. L'API de scoring Django REST exigée par le sujet est implémentée au WP4 et possède ses propres scénarios pytest.

## WP5 — Explicabilité du modèle avec SHAP

- Le service SHAP utilise Permutation SHAP sur les colonnes d'entrée du pipeline et explique la probabilité d'une classe. Il fournit la décomposition locale et vérifie le résidu d'additivité; ces contributions ne doivent pas être interprétées comme causales.
- Le chargement du CSV de référence est configurable par PROJECT35_SHAP_BACKGROUND; l'échantillonnage est déterministe et le service met en cache le résultat jusqu'à modification du fichier.
- POST /api/explain/ retourne les contributions par feature et un waterfall PNG Base64; GET /api/explain/global/ retourne mean(|SHAP|) et le graphique global. Les deux routes figurent dans Swagger.
- La référence contient uniquement les colonnes de features, correspond aux types d'entraînement et vient du train ou d'une population approuvée. Ne jamais utiliser le test final. Sans référence, l'API signale la configuration manquante.
- SHAP_MAX_BACKGROUND_ROWS et SHAP_MAX_GLOBAL_ROWS bornent le temps et la mémoire; Compose monte data/ en lecture seule.
- tests/explainability/test_shap_service.py couvre l'additivité, l'importance globale, les graphiques PNG et les incompatibilités de schéma. tests/api/test_django_scoring_api.py couvre le contrat HTTP.
- Guide complet: docs/wp5-shap.md. Les tests utilisent un classifieur synthétique; la validation métier reste suspendue à la livraison du dataset réel et du contrat cible.

## WP6 — Surveillance PSI

- Le module src/project35/monitoring/psi.py ajuste ses bins numériques quantiles et catégories connues uniquement depuis une référence stable; il traite missing, OTHER, underflow/overflow et calcule le PSI avec epsilon.
- POST /api/monitoring/psi/ produit un rapport par feature avec distributions, contributions, seuils, fenêtres et hashes; scripts/monitor_psi.py exécute le même calcul en batch et écrit un rapport JSON.
- La référence est configurée par PROJECT35_PSI_REFERENCE; PSI_BINS, PSI_EPSILON, PSI_WATCH_THRESHOLD et PSI_DRIFT_THRESHOLD règlent le calcul. Les seuils 0.10/0.25 restent indicatifs à confirmer.
- La collection Postman `postman/Project35_WP1-WP7.postman_collection.json` inclut les routes HTTP disponibles; le gate WP7 est un workflow CLI. `tests/monitoring/test_psi.py` et les tests Django couvrent la dérive synthétique et le contrat HTTP.
- Le profil et les alertes sont implémentés; référence métier et seuils restent à confirmer. Un rapport `drift` valide peut déclencher WP7; voir [docs/wp7-gate.md](wp7-gate.md).

## WP7 — Retraining, challenger et promotion gouvernée

Le workflow WP7 est décrit dans [docs/wp7-gate.md](wp7-gate.md). `scripts/run_challenger_workflow.py` entraîne le candidat configuré, l'enregistre sur `@challenger`, le compare au champion sur le même holdout labellisé et écrit le rapport de décision dans MLflow et `reports/gates/`. Seuls des deltas strictement supérieurs aux deux marges configurées permettent de déplacer `@champion`; un rapport absent, invalide ou FAIL est refusé par `registry/service.py`.

Les estimateurs actuellement sélectionnables sont Logistic Regression, Random Forest, Extra Trees et Decision Tree. Les transformations demeurent dans le Pipeline sklearn, et les paramètres du classifieur sont journalisés dans MLflow. Le déclenchement est manuel ou par passage explicite d'un rapport PSI `drift` avec alerte; il n'y a pas d'ordonnanceur automatique.

Configuration et exécution:

```powershell
Copy-Item configs/gate.example.yaml configs/gate.yaml
pipenv run python scripts/run_challenger_workflow.py --data-config configs/data_contract.yaml --gate-config configs/gate.yaml --mlflow-config configs/mlflow.yaml
pipenv run pytest tests/gating tests/training/test_tracking.py tests/data/test_phase1.py -q
```

Les tests gate utilisent des données synthétiques. La validation métier, les marges officielles et toute promotion réelle restent en attente du dataset et des décisions approuvées. WP8 doit intégrer le workflow à la démonstration Compose et fournir les preuves d'exécution réelle.

## WP8 — Démonstration et remise

`docker-compose.yml` fournit maintenant MLflow avec stockage persistant, FastAPI/Swagger, Django REST et un profil de commande challenger; `.github/workflows/ci.yml` lance pytest, Django `check` et la validation de Compose. `scripts/check_stack.py` vérifie health et OpenAPI des services actifs. Le guide [docs/wp8-delivery.md](wp8-delivery.md) détaille le démarrage, les endpoints, le protocole de démonstration, les commandes QA et les preuves à archiver. Un plan indicatif de 20 diapositives est proposé dans [presentation/outline-wp8.md](../presentation/outline-wp8.md), à faire confirmer par l'enseignant.

Les configurations Compose, health/Swagger HTTP, le contrôle Django et la suite Pytest ont été exécutés dans le conteneur Python 3.11 avec les sources courantes; dernier résultat consigné : 50 tests passés, 77% de couverture. Les exécutions MLOps et l'archive disponibles portent exclusivement sur le dataset synthétique approuvé; elles ne remplacent pas les résultats métier ou officiels du cours.
