# WP8 — Intégration, démonstration et remise

## Périmètre et état

WP8 rassemble les lots précédents dans un environnement relançable: MLflow (Tracking/Registry/artifacts), FastAPI (contrat), Django REST (scoring, SHAP, PSI), les tests pytest, le client Postman et les éléments de démonstration. Le Compose a été construit et démarré; health/OpenAPI de ses services ont passé le smoke-check. La validation sur un jeu approuvé et les captures métier restent à produire.

Le dataset métier du cours et son approbation restent absents; un jeu synthétique réservé aux smoke tests est présent. Les CLI phase1/train/challenger bloquent le fit métier tant que la fiche de provenance réelle n'est pas approuvée et que son SHA-256 ne correspond pas au CSV.

## Services Compose

| Service | Port | Fonction | Stockage |
|---|---:|---|---|
| `mlflow` | 5000 | Tracking UI/REST, Registry et proxy d'artifacts | volume `mlflow_data` pour SQLite et artifacts |
| `api` | 8000 | FastAPI, validation du contrat de données et Swagger | code/config/données montés en développement |
| `django-api` | 8001 | scoring champion, SHAP et PSI | volume `django_data` pour SQLite; données montées en lecture seule |
| `synthetic-demo-api` | 8002 | instance optionnelle réservée au modèle synthétique et aux appels SHAP/PSI de démo | profil `synthetic`, volume distinct `django_synthetic_data`; n'écrit pas dans la base Django métier |
| `challenger-workflow` | — | profil ponctuel pour entraînement et gate WP7 | config/données en lecture seule; rapports persistés dans `./reports` |

MLflow est un service de développement local, avec SQLite. Pour un déploiement partagé, utiliser un backend SQL serveur, un artifact store durable externe, authentification, TLS et sauvegardes. Le serveur monté dans Compose n'est pas un déploiement de production.

## Démarrage reproductible

Prérequis: Docker Engine avec Compose v2. Pour les opérations de modèle, renseigner les fichiers locaux ignorés/non committés et placer les données autorisées sous `data/`.

```powershell
Copy-Item configs/data_contract.example.yaml configs/data_contract.yaml
Copy-Item configs/mlflow.example.yaml configs/mlflow.yaml
Copy-Item configs/gate.example.yaml configs/gate.yaml
docker compose config --quiet
docker compose up --build -d
docker compose ps
# Démarrer séparément la démo synthétique, sans changer la config métier sur 8001
docker compose --profile synthetic up -d synthetic-demo-api
```

Interfaces:

- FastAPI Swagger: `http://127.0.0.1:8000/docs`
- Django scoring/SHAP/PSI Swagger: `http://127.0.0.1:8001/api/docs/`
- API de démonstration synthétique (modèle `project35-synthetic-churn`): `http://127.0.0.1:8002/api/docs/`
- MLflow UI: `http://127.0.0.1:5000`
- Health: `http://127.0.0.1:8000/health/`, `http://127.0.0.1:8001/health/`, `http://127.0.0.1:5000/health`

Django métier sur 8001 démarre en état `degraded` tant qu'un alias métier n'est pas configuré. Cela n'empêche pas l'instance isolée sur 8002 de charger le champion synthétique version 1. Après la première requête, vérifier `model_loaded=true` sur le service ciblé.

La collection Postman `postman/Project35_WP1-WP7.postman_collection.json` utilise les valeurs synthétiques et pointe vers 8002 pour Django. Elle couvre health, contrat FastAPI, modèle, prédiction, OpenAPI, SHAP local/global et PSI. Le gate WP7 reste un workflow CLI.

## Démonstration par exigence

1. **Contrat et données:** exécuter `pipenv run pytest tests/data tests/api/test_data_contract_api.py`; appeler validation FastAPI avec un petit payload synthétique; présenter le rapport qualité et le contrat approuvé sans montrer des lignes personnelles.
2. **MLflow et modèle:** lancer l'entraînement avec `scripts/train.py`, inspecter paramètres, hash, signature, rapport et graphiques sous MLflow; enregistrer la première version comme champion avec `scripts/register_model.py`. Les données de démonstration doivent être celles approuvées par l'encadrant.
3. **Serving:** utiliser la collection Postman, vérifier `/health/`, `/api/model-info/`, puis `/api/predict/` avec les features exactes du modèle. Capturer version alias, request ID et statut, sans exposer les valeurs personnelles.
4. **SHAP:** configurer la variable `PROJECT35_SHAP_BACKGROUND` vers un CSV autorisé (dans Compose, chemin `/app/data/processed/shap_reference.csv`); appeler l'explication locale et globale, conserver les graphiques et expliquer que les contributions ne sont pas causales.
5. **PSI:** configurer la référence approuvée et exécuter `scripts/monitor_psi.py` ou l'endpoint; conserver hash, fenêtre, PSI/features et statut. Seuils de démonstration doivent être présentés comme indicatifs jusqu'à confirmation métier.
6. **WP7 gate:** entraîner/enregistrer/évaluer le challenger via `scripts/run_challenger_workflow.py`; tester un cas PASS et un cas FAIL. Prouver que FAIL conserve `@champion`, qu'un rapport PASS déplacera l'alias, puis appeler `/health/` et `/api/model-info/` pour montrer la version rechargée. Le holdout doit être indépendant et exclu de l'entraînement; marges fixées avant observation.
7. **Qualité de livraison:** lancer la suite complète et les vérifications Compose/Django ci-dessous; joindre le résumé, les avertissements et les limites, sans remplacer un résultat manquant par une capture illustrative.

## Batterie de vérification

```powershell
pipenv run pytest --cov=project35 --cov-report=term-missing
pipenv run python django_app/manage.py check
docker compose config --quiet
docker compose ps
pipenv run python scripts/check_stack.py
pipenv run python scripts/preflight_data.py --config configs/data_contract.yaml --provenance configs/data_provenance.yaml --require-approved
```

La CI GitHub Actions exécute pytest/couverture, `manage.py check` et l'analyse du fichier Compose. Une configuration Compose valide n'établit pas que les images démarrent; conserver aussi le résultat de `docker compose ps` et les appels HTTP réels lors de la démonstration.

## Vérification effectuée dans l'environnement Compose

Le 20 novembre 2026, `docker compose config --quiet` a terminé avec code 0 (Docker a averti qu'il ne pouvait pas lire `C:\Users\Amine\.docker\config.json`; cela n'a pas empêché Compose). `docker compose up --build -d` a construit les images et lancé MLflow, FastAPI et Django. Le smoke-check réseau a réussi sur FastAPI health/OpenAPI, Django health/OpenAPI et MLflow health. Django a répondu `degraded`, `model_loaded=false`, ce qui est attendu en l'absence d'un modèle champion métier enregistré. `manage.py check` a signalé zéro problème.

Vérification WP8 du 20 novembre 2026 : suite complète **48 passed**, couverture de `project35` **76%**, en 49,35 s. Les cinq avertissements restants viennent des compatibilités Starlette/httpx, SHAP/matplotlib et SQLAlchemy/MLflow. `manage.py check` est propre; `docker compose config --quiet` réussit; le smoke checker sur le réseau Compose rapporte FastAPI/Django OpenAPI et MLflow health OK, avec Django métier `degraded` attendu sans modèle métier. Les payloads de la collection Postman ont été rejoués par HTTP sur le profil synthétique: contrat, prédiction, SHAP local/global et PSI. Le score PSI du petit payload Postman déclenche une alerte sur seulement deux observations; c'est une démonstration du endpoint, pas une alerte opérationnelle. Les résumés d'exécution sont dans `reports/synthetic/wp8-smoke.json` (ce dossier est ignoré par Git). Les rapports synthétiques n'évaluent pas les performances métier.

La collection Postman JSON est valide et ses variables `sample_instance` et `psi_records` suivent les 12 features du modèle. Le profil `synthetic` monte les CSV en lecture seule, utilise un volume Django dédié et charge exclusivement `project35-synthetic-churn`; la base métier de 8001 et son volume ne sont pas partagés. Les fichiers de preuve sont dans `reports/synthetic/` (ignorés par Git).

## Preuves à remettre

Créer un dossier daté, par exemple `reports/delivery/2026-11-20/`, avec:

- versions/empreintes des données autorisées, provenance/licence, contrat final et marges approuvées;
- résumés d'exécution pytest/CI, `manage.py check`, configuration et état Compose;
- run IDs, model versions, alias avant/après, hash du holdout et rapports gate PASS/FAIL;
- résultats non identifiants des endpoints, captures Swagger/Postman et graphiques SHAP/PSI pertinents;
- inventaire des limites restantes et instructions de relance.

Ne pas inclure les CSV bruts, secrets, tokens, bases SQLite, modèles protégés ou artefacts volumineux interdits. La collection [Postman WP1–WP7](../postman/Project35_WP1-WP7.postman_collection.json) couvre les endpoints HTTP; le gate WP7 s'exécute comme CLI et son audit est dans MLflow/JSON, pas dans une route HTTP.

## Présentation et clôture

Le plan indicatif de présentation est dans [presentation/outline-wp8.md](../presentation/outline-wp8.md). Confirmer le nombre de diapositives, le contenu attendu et le dataset avec l'enseignant: le sujet fourni ne spécifie pas ces décisions. La remise finale reste conditionnée à l'exécution sur données autorisées, à l'approbation des seuils et à la production des preuves listées ci-dessus.

## Vérification complémentaire — 2026-11-21

Pytest a été relancé dans un conteneur temporaire Python 3.11 avec les dossiers `scripts/`, `tests/`, `django_app/` et `src/` de la copie courante montés en lecture seule : après test d'intégration API TreeExplainer, **50 passed**, couverture `project35` **77%**, durée **61,43 s**, **5 avertissements** de dépendances. Les caches pytest et bytecode hôte ont été désactivés pour ce run. Le code testé correspond donc aux fichiers du workspace, et pas uniquement aux anciens fichiers incorporés à l'image.

`python django_app/manage.py check` dans un conteneur temporaire a répondu `System check identified no issues (0 silenced)`. `docker compose config --quiet` et `docker compose --profile synthetic config --quiet` se terminent sans erreur; Docker affiche un avertissement d'accès au fichier utilisateur `C:\Users\Amine\.docker\config.json`, sans bloquer la validation de configuration. Le health HTTP courant indique FastAPI `ok`, Django synthétique `ok` avec `project35-synthetic-churn@champion` v1, et MLflow `/health` `200`.

Une instance Django temporaire sur `8003`, avec le code courant et le modèle synthétique LR retenu, a servi les endpoints SHAP local/global en HTTP 200. Les deux réponses indiquent `PermutationExplainer`; la locale présente 12 features et un résidu d'additivité `-5.55e-17`. Le test API intégré entraîne un pipeline Decision Tree mixte, appelle `/api/explain/` et `/api/explain/global/`, et vérifie `TreeExplainer`, les features source et l'additivité. L'instance temporaire a été arrêtée avec son conteneur `--rm`; les services habituels n'ont pas été remplacés.

Un paquet technique synthétique a été construit à `deliverables/Project35-synthetic-technical-demo-2026-11-21.zip`. Inspection : 84 chemins d'entrée uniques, aucun CSV brut, base de données, sérialisation de modèle, fichier `.env` ou bytecode inclus. Le paquet rassemble les sources, tests, configurations synthétiques, Postman, documentation et rapports JSON synthétiques; il ne constitue pas l'archive finale du cours avant confirmation de son format.
