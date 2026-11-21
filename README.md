# Project 35 — MLOps end-to-end

## État actuel

Les WP1–WP7 sont implémentés et exercés sur un profil synthétique isolé; WP8 fournit le contrôle de stack et les outils de démonstration. Le dataset, la cible et les seuils officiels du cours restent à confirmer avant toute validation métier. Les données synthétiques servent uniquement à vérifier les mécanismes et ne sont pas des résultats du projet métier.

FastAPI/Swagger permet de vérifier le contrat dès le démarrage. L'API de scoring Django REST du WP4 sert le champion MLflow; les deux surfaces sont testées avec pytest.

## Installation locale

```powershell
pipenv install --dev
```

## Dataset, provenance et pré-validation

Le dataset métier n'est pas fourni. Copier les exemples de contrat/provenance, remplir uniquement après approbation, puis inspecter avant tout entraînement:

```powershell
Copy-Item configs/data_contract.example.yaml configs/data_contract.yaml
Copy-Item configs/data_provenance.example.yaml configs/data_provenance.yaml
pipenv run python scripts/preflight_data.py --config configs/data_contract.yaml --provenance configs/data_provenance.yaml
```

Après validation par le data owner, inscrire `approval_status: approved` et le SHA-256 retourné dans la provenance, puis relancer avec `--require-approved`. Les commandes phase1, train et challenger appliquent cette garde avant le fit. Guide: [docs/dataset-readiness.md](docs/dataset-readiness.md).

Un jeu B2B churn **synthétique uniquement** est inclus pour tester la plomberie: 4 000 lignes de fit et 1 000 lignes de holdout indépendant, dans `data/synthetic/`, avec configurations dédiées `configs/*.synthetic.yaml`. Recréer et pré-valider:

```powershell
pipenv run python scripts/generate_synthetic_dataset.py
pipenv run python scripts/preflight_data.py --config configs/data_contract.synthetic.yaml --provenance configs/data_provenance.synthetic.yaml --require-approved
```

Les métriques obtenues sur ce jeu illustratif ne représentent pas les résultats métier ou les données officielles du cours.

Pour l'entraîner et l'enregistrer sous un modèle séparé du modèle métier, démarrer Compose puis suivre le parcours détaillé dans [docs/dataset-readiness.md](docs/dataset-readiness.md). Le profil synthétique MLflow utilise `project35-synthetic-churn`.

## Tests, Swagger et phase 1

```powershell
pipenv run pytest --cov=project35 --cov-report=term-missing
pipenv run uvicorn project35.api.main:app --reload --app-dir src
```

Ouvrir `http://127.0.0.1:8000/docs` pour Swagger et `/openapi.json` pour le contrat OpenAPI.

## WP2 — MLflow Tracking

Après avoir renseigné le dataset officiel dans `configs/data_contract.yaml` :

```powershell
Copy-Item configs/mlflow.example.yaml configs/mlflow.yaml
pipenv run mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000
```

Dans un second terminal, après configuration du dataset officiel :

```powershell
pipenv run python scripts/train.py --data-config configs/data_contract.yaml --mlflow-config configs/mlflow.yaml
```

Le run trace les métriques de validation, l'empreinte SHA-256 du dataset, les paramètres, versions logicielles, signature du pipeline, exemple d'entrée, rapport et graphiques. Le jeu de test final n'est pas utilisé pour sélectionner la baseline.

## WP3 — Model Registry

Copier le `run_id` retourné par l'entraînement et enregistrer la première version :

```powershell
pipenv run python scripts/register_model.py --run-id <RUN_ID> --model-name project35-classifier --alias champion
```

Pour un candidat ultérieur, utiliser `--alias challenger`. Le déplacement d'un `@champion` existant exige un rapport gate PASS persisté que le registre vérifie (WP7). Voir [docs/phase-1.md](docs/phase-1.md) pour la résolution d'alias et les livrables.

## WP4 — API de scoring Django REST

Démarrer d'abord MLflow, puis le scoring API :

```powershell
pipenv run python django_app/manage.py migrate --noinput
pipenv run python django_app/manage.py runserver 127.0.0.1:8001
```

Swagger: `http://127.0.0.1:8001/api/docs/`; OpenAPI: `http://127.0.0.1:8001/api/schema/`.

Routes principales: `GET /health/`, `POST /api/predict/` et `GET /api/model-info/`. Exemple de prédiction:

```json
{"instances": [{"feature_1": 12.4, "feature_2": "A"}]}
```

Remplacer les features d'exemple par les colonnes exactes du modèle enregistré. Les tests pytest mockent le registre et utilisent un faux classifieur; ils vérifient le contrat HTTP sans dépendre d'un serveur MLflow actif.

Pour lancer le pipeline sur le dataset officiel, copier `configs/data_contract.example.yaml` vers `configs/data_contract.yaml`, renseigner les valeurs confirmées, déposer le CSV au chemin configuré, puis lancer :

```powershell
python scripts/phase1.py --config configs/data_contract.yaml
```

## WP5 — Explicabilité SHAP

Le scoring Django expose `POST /api/explain/` (contributions locales, additivité et waterfall) et `GET /api/explain/global/` (mean absolute SHAP, bar summary et beeswarm). Les deux routes sont documentées dans Swagger: `http://127.0.0.1:8001/api/docs/`. Les visualisations PNG sont encodées en Base64 dans la réponse JSON.

Configurer `PROJECT35_SHAP_BACKGROUND` vers un CSV de référence contenant exactement les features d'entrée, sans cible ni identifiant. Utiliser les données de train ou une population de référence approuvée, jamais le test final. En local:

```powershell
$env:PROJECT35_SHAP_BACKGROUND = "data/processed/shap_reference.csv"
pipenv run python django_app/manage.py runserver 127.0.0.1:8001
```

Compose monte `data/` en lecture seule et utilise par défaut `/app/data/processed/shap_reference.csv`. Les budgets sont configurables par `SHAP_MAX_BACKGROUND_ROWS` (40) et `SHAP_MAX_GLOBAL_ROWS` (50). SHAP explique les contributions au modèle et ne constitue pas une explication causale. Voir [docs/wp5-shap.md](docs/wp5-shap.md).

Tests WP5: `pipenv run pytest tests/explainability tests/api/test_django_scoring_api.py`.

Client Postman HTTP: [postman/Project35_WP1-WP7.postman_collection.json](postman/Project35_WP1-WP7.postman_collection.json) (WP7 gate reste une commande CLI); revue des exigences WP0–WP8: [docs/revue-conformite-wp0-wp8.md](docs/revue-conformite-wp0-wp8.md).

## WP6 — Monitoring PSI

Django REST expose `POST /api/monitoring/psi/` pour comparer un batch courant à la référence configurée par `PROJECT35_PSI_REFERENCE`. Les coupures numériques et catégories sont apprises sur la référence puis réutilisées; catégories inconnues, valeurs manquantes et valeurs hors plage sont suivies séparément. Le rapport donne le PSI par feature, les bins, status/alerte et les SHA-256 des populations. Les seuils 0.10/0.25 sont indicatifs, à confirmer avec le sujet.

Commande batch: `pipenv run python scripts/monitor_psi.py --reference data/processed/psi_reference.csv --current data/current/batch.csv --window batch-001 --output reports/monitoring/psi_report.json`. Swagger: `http://127.0.0.1:8001/api/docs/`; collection: [Postman WP1-WP7](postman/Project35_WP1-WP7.postman_collection.json). Guide: [docs/wp6-psi.md](docs/wp6-psi.md).

## WP7 — Retraining et gate champion/challenger

Le workflow [scripts/run_challenger_workflow.py](scripts/run_challenger_workflow.py) entraîne un candidat, met à jour `@challenger`, le compare au champion sur un holdout labellisé indépendant, puis ne promeut que si F1 et ROC-AUC dépassent toutes deux les marges configurées. Le rapport horodaté et hashé est conservé dans MLflow et `reports/gates/`; une dérive PSI peut déclencher cette commande avec un rapport valide, mais ne constitue pas une autorisation de promotion.

```powershell
Copy-Item configs/gate.example.yaml configs/gate.yaml
# Renseigner les configs data, MLflow et holdout approuvés
pipenv run python scripts/run_challenger_workflow.py --data-config configs/data_contract.yaml --gate-config configs/gate.yaml --mlflow-config configs/mlflow.yaml
pipenv run pytest tests/gating tests/training/test_tracking.py tests/data/test_phase1.py -q
```

Règles, fichiers, limites et déroulé: [docs/wp7-gate.md](docs/wp7-gate.md). Le gate est couvert par tests synthétiques; aucune promotion sur données métier n'est revendiquée. Le dataset, la cible et les marges restent à confirmer. WP8 couvre la démonstration complète, Compose et les preuves de remise.

## Docker Compose

```powershell
docker compose config --quiet
docker compose up --build -d
docker compose ps
```

Compose démarre MLflow persistant (5000), FastAPI/Swagger (8000) et Django REST (8001). Les interfaces sont sur `http://127.0.0.1:5000`, `http://127.0.0.1:8000/docs` et `http://127.0.0.1:8001/api/docs/`. Django peut afficher `degraded` jusqu'à l'initialisation de `@champion`. Le profil ponctuel `challenger-workflow` demande des fichiers data/gate locaux: `docker compose --profile workflow run --rm challenger-workflow`.

Pour la démo synthétique isolée (MLflow `project35-synthetic-churn`, données SHAP/PSI de démonstration, base Django séparée et port 8002), démarrer `docker compose --profile synthetic up -d synthetic-demo-api`; ouvrir `http://127.0.0.1:8002/api/docs/`. La collection Postman préconfigure FastAPI 8000, Django synthétique 8002 et MLflow 5000. Arrêter seulement ce service avec `docker compose --profile synthetic stop synthetic-demo-api`; ne pas utiliser `down -v` si l'on veut préserver les volumes.

Le plan de livraison et les preuves sont décrits dans [docs/wp8-delivery.md](docs/wp8-delivery.md); le plan de présentation est dans [presentation/outline-wp8.md](presentation/outline-wp8.md). Au 21 novembre 2026, la suite complète passe **50 tests**, couverture **77%**; `manage.py check`, les configurations Compose et les principaux endpoints HTTP ont été vérifiés. La démonstration reste synthétique et isolée du service métier. Les runs, limites et résultats sont consignés dans [docs/synthetic-demo.md](docs/synthetic-demo.md). Le plan de 20 diapositives et l'archive technique sont produits; aucune métrique synthétique ne constitue un résultat métier ou officiel du cours.

La validation de données synthétiques du 21 novembre et le statut des neuf tâches techniques sont consignés dans [docs/dataset-validation-2026-11-21.md](docs/dataset-validation-2026-11-21.md) et [docs/remaining-delivery-tasks.md](docs/remaining-delivery-tasks.md). Elles sont terminées dans le périmètre synthétique approuvé; une validation officielle nécessiterait encore le dataset, les seuils et le format demandés par le cours.

Les vérifications WP1–WP4 et leurs limites actuelles sont consignées dans [docs/wp1-wp4-verification.md](docs/wp1-wp4-verification.md).
