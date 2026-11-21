# Rapport d'audit fichier par fichier — Project 35

**Audit et nettoyage des caches : 2026-11-21.** Inventaire initial post-nettoyage : 103 fichiers accessibles hors métadonnées Git; le présent rapport constitue le 104e fichier. Le code applicatif, les points d'entrée, les configurations, la base Django, les données et les rapports n'ont pas été supprimés ni déplacés.

## Résultat du nettoyage

- Retirés : `.coverage`; les cinq fichiers de `.pytest_cache/`; 29 fichiers `.pyc` et leurs dix dossiers `__pycache__/`.
- Vérifié après nettoyage : `.coverage` et `.pytest_cache/` absents; zéro `.pyc` accessible; les fichiers Python source correspondants existent toujours.
- Aucun doublon exact d'empreinte SHA-256 parmi les fichiers accessibles. Aucun `.bak`, `.old`, `.tmp`, `copy*`, `test_scratch*` ou fichier `.log` trouvé. Aucun déplacement ni mise en quarantaine.
- Les configurations Pipenv et `pyproject.toml` ont des champs communs mais servent deux flux (environnement Pipenv et packaging/pytest) : elles sont conservées.
- Les rapports JSON sont des exécutions distinctes avec identifiants/horodatages; même lorsqu'ils montrent le même cas de gate, ils ne sont pas des copies octet pour octet et sont conservés comme preuves d'audit.

## Fichiers conservés

### Racine et CI

| Fichier | Rôle, fonctionnement, interactions | Importance |
|---|---|---|
| `.gitignore` | Exclut bases SQLite, secrets `.env`, CSV de données privées, rapports, artefacts MLflow, caches Python et couverture. Contrôle ce qui peut entrer dans Git; ne doit pas être supprimé. | Configuration critique |
| `docker-compose.yml` | Définit MLflow, FastAPI, Django métier, profil Django synthétique et workflow challenger; configure ports, volumes, variables, dépendances et health checks. Lance l'application dans Docker. | Critique / exécution |
| `Dockerfile` | Construit l'image Python 3.11, copie métadonnées, `src`, `scripts`, `django_app` et installe le projet avec ses extras de test; image utilisée par Compose. | Critique / build |
| `Pipfile` | Déclare les dépendances d'exécution et de développement pour Pipenv, Python 3.11. | Configuration |
| `Pipfile.lock` | Fige les versions résolues par Pipenv; consommé lors de la synchronisation reproductible de l'environnement. | Configuration critique |
| `pyproject.toml` | Décrit le package, ses dépendances, découverte sous `src`, configuration pytest/pytest-django et couverture. Utilisé par pip, Docker et pytest. | Configuration critique |
| `README.md` | Point d'entrée de la documentation : installation, contrats, commandes, WP1–WP8, API et limites synthétiques; renvoie vers scripts/configurations/docs. | Support essentiel |
| `.github/workflows/ci.yml` | Pipeline CI GitHub Actions; installe le projet, exécute les contrôles et tests définis pour prévenir les régressions. | Configuration critique |

### Configurations (`configs/`)

| Fichier | Rôle, fonctionnement, interactions | Importance |
|---|---|---|
| `configs/data_contract.example.yaml` | Modèle à compléter avec chemin, cible, classe positive, colonnes, split et seed du dataset officiel; exemple/documentation, pas contrat de données actif. | Configuration |
| `configs/data_contract.synthetic.yaml` | Contrat approuvé pour le dataset synthétique : 12 features, cible `churn`, classe positive 1, split stratifié et seed 35; chargé par preflight et entraînement de démonstration. | Configuration / données |
| `configs/data_provenance.example.yaml` | Gabarit de source, licence, version, hash, approbateur et usage officiel; reste `pending` tant qu'il n'est pas remplacé par des métadonnées réelles approuvées. | Configuration / gouvernance |
| `configs/data_provenance.synthetic.yaml` | Provenance et périmètre autorisé du générateur synthétique; le preflight compare son SHA-256 au dataset et requiert l'état approuvé pour entraîner en démonstration. | Configuration / gouvernance |
| `configs/gate.example.yaml` | Politique exemple du gate pour un futur holdout approuvé; précise champion, modèle challenger, hyperparamètres, marges et dossier de rapport. | Configuration |
| `configs/gate.synthetic.yaml` | Politique d'évaluation synthétique utilisée par le workflow WP7; exige des améliorations strictement positives F1 et ROC-AUC et écrit sous `reports/synthetic-gates/`. | Configuration / MLOps |
| `configs/mlflow.example.yaml` | URI, expérience et nom de modèle du futur environnement MLflow métier. | Configuration |
| `configs/mlflow.synthetic-compose.yaml` | URI MLflow joignable depuis le réseau Compose (`mlflow:5000`), nom de l'expérience et modèle réservés aux démonstrations. | Configuration / exécution |
| `configs/mlflow.synthetic.yaml` | URI MLflow locale (`127.0.0.1:5000`) et noms d'expérience/modèle synthétiques pour les commandes sur l'hôte. | Configuration |

### Données et instructions (`data/`)

| Fichier | Rôle, fonctionnement, interactions | Importance |
|---|---|---|
| `data/README.md` | Règles de dépôt local : autorisation, provenance, licence, protection des CSV et preflight avant entraînement. | Support / gouvernance |
| `data/current/README.md` | Explique l'emplacement des fenêtres courantes PSI et les métadonnées de période à tracer. | Support |
| `data/holdout/README.md` | Définit le holdout indépendant du gate et interdit son utilisation pour fit, tuning ou seuils; les fichiers de données sont ignorés par Git. | Support / gouvernance |
| `data/processed/README.md` | Décrit les données dérivées et références SHAP/PSI locales, leur provenance et hash. | Support |
| `data/raw/README.md` | Décrit l'emplacement du dataset source non transformé et sa protection hors Git. | Support |
| `data/synthetic/README.md` | Documente le caractère généré, les populations, les limites et les usages SHAP/PSI/gate; relié au générateur et au manifeste. | Support / gouvernance |
| `data/synthetic/manifest.json` | Inventorie les CSV synthétiques, tailles, classes, hashes, seed, populations indépendantes et limites; vérifié par les contrôles de dataset. | Données / traçabilité |
| `data/synthetic/b2b_churn_train.csv` | 4 000 observations synthétiques, 12 features et cible `churn`; consommé par preflight, splits et entraînement des candidats. | Données / démonstration |
| `data/synthetic/b2b_churn_gate_evaluation.csv` | Population synthétique indépendante de 1 000 lignes avec cible; réservée à l'évaluation WP7 et jamais au fit. | Données / évaluation |
| `data/synthetic/b2b_churn_shap_reference.csv` | Référence de 100 lignes sans cible, avec les features seulement; entrée du service SHAP dans la démo Django synthétique. | Données / explicabilité |
| `data/synthetic/b2b_churn_psi_reference.csv` | Référence PSI de 4 000 lignes train sans cible; entrée de `PSIProfile` et de l'endpoint de monitoring. | Données / monitoring |
| `data/synthetic/b2b_churn_psi_stable_batch.csv` | Batch indépendant synthétique sans cible, utilisé pour tester la branche PSI stable et l'API/CLI. | Données / test fonctionnel |
| `data/synthetic/b2b_churn_psi_drift_batch.csv` | Batch de démonstration décalé artificiellement en région et usage produit; vérifie l'alerte PSI, sans représenter une dérive réelle. | Données / test fonctionnel |

### Application Django (`django_app/`)

| Fichier | Rôle, fonctionnement, interactions | Importance |
|---|---|---|
| `django_app/manage.py` | Point d'entrée des commandes Django (`check`, migrations, serveur); charge `config.settings`. | Critique / entrée |
| `django_app/db.sqlite3` | Base SQLite locale de l'instance Django existante; préservée car fichier d'état d'exécution, même si une DB dédiée existe en Compose. | Critique / données d'exécution |
| `django_app/config/__init__.py` | Marque le package Django `config`; importé par Django/Python. | Critique / support |
| `django_app/config/settings.py` | Configure apps, middleware, REST Framework, drf-spectacular, SQLite, variables MLflow, CORS/hosts et réglages SHAP/PSI. Chargé par manage.py et WSGI/ASGI. | Critique / configuration |
| `django_app/config/urls.py` | Route health, scoring, model-info, SHAP, PSI, schéma OpenAPI et Swagger vers les vues/serializers. | Critique / API |
| `django_app/config/wsgi.py` | Expose l'application Django au serveur WSGI. | Critique / exécution |
| `django_app/config/asgi.py` | Expose l'application Django au serveur ASGI. | Critique / exécution |
| `django_app/predictions/__init__.py` | Marque l'application predictions comme package Python. | Cœur métier / support |
| `django_app/predictions/apps.py` | Déclare `PredictionsConfig`, nom et configuration de l'application Django. | Configuration |
| `django_app/predictions/serializers.py` | Contrats REST de prédiction, réponse, health et métadonnées du modèle; utilisés par les vues et le schéma OpenAPI. | Cœur métier / API |
| `django_app/predictions/explain_serializers.py` | Valide les entrées d'explication et décrit réponses locales/globales, méthode SHAP, résidu et PNG pour Swagger. | Cœur métier / API |
| `django_app/predictions/monitoring_serializers.py` | Valide les batchs PSI et sérialise seuils, résultats par variable, hashes et alertes. | Cœur métier / API |
| `django_app/predictions/services.py` | Provider paresseux de champion MLflow; résout alias/version, charge modèle, gère cache et erreurs de disponibilité. Appelé par health/predict/explain views. | Cœur métier |
| `django_app/predictions/explainability.py` | Charge/cache la référence SHAP d'après chemin, date/tailles du fichier et hash, puis construit `ShapExplanationService`. | Cœur métier |
| `django_app/predictions/monitoring.py` | Charge/cache le profil PSI d'après la référence configurée et expose l'instance aux vues. | Cœur métier |
| `django_app/predictions/views.py` | Implémente health, predict, model-info, explication locale/globale et monitoring PSI; délègue aux providers/services et renvoie erreurs REST contrôlées. | Cœur métier / API |

### Package MLOps (`src/project35/`)

| Fichier | Rôle, fonctionnement, interactions | Importance |
|---|---|---|
| `src/project35/__init__.py` | Marque le package installable `project35`. | Cœur métier / support |
| `src/project35/api/__init__.py` | Marque le sous-package API. | Support |
| `src/project35/api/main.py` | Application FastAPI : health, informations du contrat et validation batch; pydantic contrôle les requêtes et les fonctions data génèrent le rapport qualité/OpenAPI. | Cœur métier / API |
| `src/project35/data/__init__.py` | Marque le sous-package données. | Support |
| `src/project35/data/contract.py` | Lit YAML/CSV, matérialise `DataContract`, vérifie colonnes, cible, types/valeurs/duplications et fournit les erreurs métier réutilisées par preflight, phase1 et entraînement. | Cœur métier |
| `src/project35/data/phase1.py` | Crée splits train/validation/test sans cible dans X; gère split stratifié/chronologique et pipeline sklearn numérique/catégoriel pour LR, arbres, RF et ExtraTrees. | Cœur métier / ML |
| `src/project35/training/__init__.py` | Marque le sous-package entraînement. | Support |
| `src/project35/training/trainer.py` | Ajuste le modèle train, mesure validation, produit courbes/rapport/signature, trace params/métriques/tags/artefacts et pipeline dans MLflow. Ne choisit pas à partir du test final. | Cœur métier / ML-MLOps |
| `src/project35/registry/__init__.py` | Marque le sous-package registry. | Support |
| `src/project35/registry/service.py` | Enregistre des runs, assigne/résout alias `champion`/`challenger`, charge les modèles et impose rapport gate PASS vérifié avant déplacement du champion. | Cœur métier / MLOps |
| `src/project35/explainability/__init__.py` | Marque le sous-package explainability. | Support |
| `src/project35/explainability/shap_service.py` | Calcule SHAP local/global : TreeExplainer interventional pour pipelines arbre avec regroupement one-hot par feature source; PermutationExplainer pour modèles non-arbre; vérifie additivité et génère PNG. | Cœur métier / ML |
| `src/project35/monitoring/__init__.py` | Marque le sous-package monitoring. | Support |
| `src/project35/monitoring/psi.py` | Construit bins de référence figés, calcule PSI par feature, gère valeurs manquantes/catégories inédites et classe stable/watch/drift selon seuils. | Cœur métier / monitoring |
| `src/project35/gating/__init__.py` | Marque le sous-package gate. | Support |
| `src/project35/gating/service.py` | Charge les alias MLflow, vérifie schéma et holdout, calcule F1/ROC-AUC, compare aux marges, journalise rapport/hash et applique PASS/FAIL/promotion. | Cœur métier / MLOps |

### Scripts (`scripts/`)

| Fichier | Rôle, fonctionnement, interactions | Importance |
|---|---|---|
| `scripts/__init__.py` | Marque les scripts comme package afin de permettre leurs imports croisés. | Support |
| `scripts/preflight_data.py` | CLI de provenance, hash et qualité; refuse approbation/cible/hash incohérents, produit readiness, qualité et split sans entraîner. Appelé aussi par phase1/train/workflow. | Cœur métier / gouvernance |
| `scripts/phase1.py` | Orchestre preflight puis baseline phase1 et rapport qualité; lit contrat/provenance, délègue à `run_phase1`. | Entrée ML |
| `scripts/train.py` | CLI pour entraîner LR/RF/DecisionTree/ExtraTrees sur split fixe et tracer dans MLflow; valide d'abord provenance et hash. | Entrée ML / MLOps |
| `scripts/register_model.py` | CLI d'enregistrement d'un run MLflow et assignation optionnelle d'alias, description et gate-report; délègue à registry governance. | Entrée MLOps |
| `scripts/run_challenger_workflow.py` | Orchestre preflight, entraînement challenger, enregistrement, gate et alias; peut exiger une alerte PSI valide et un holdout indépendant. | Entrée MLOps critique |
| `scripts/monitor_psi.py` | CLI de comparaison CSV référence/batch, seuils/bins/hash, puis écrit un rapport JSON et statut; appelle `PSIProfile`. | Entrée monitoring |
| `scripts/generate_synthetic_dataset.py` | Génère populations artificielles train, gate, références et batchs stable/drift avec seed, calcul de hash et manifeste; données exclusivement techniques. | Support / test de données |
| `scripts/check_stack.py` | Smoke check HTTP FastAPI health/OpenAPI, Django health/schema et MLflow health; échoue si les chemins/états requis manquent. | Support / validation |

### Tests (`tests/`)

| Fichier | Rôle, fonctionnement, interactions | Importance |
|---|---|---|
| `tests/conftest.py` | Fixtures communes de records et DataFrame utilisés par tests de contrat, split et qualité. | Test |
| `tests/api/test_data_contract_api.py` | Vérifie Swagger, health FastAPI, validation acceptée/rejetée, colonnes/cible requises et payload vide. | Test / API |
| `tests/api/test_django_scoring_api.py` | Teste routes et schéma Django; prédiction, erreurs, metadata, SHAP local/global, TreeExplainer intégré, et PSI avec références temporaires/mocks. | Test / API |
| `tests/data/test_phase1.py` | Teste rapport qualité, cibles invalides, séparation des features et fit de pipelines mixtes pour modèles autorisés. | Test / ML |
| `tests/data/test_preflight.py` | Teste hash, readiness, cible et rejet de provenance non approuvée ou dataset altéré. | Test / gouvernance |
| `tests/data/test_synthetic_dataset.py` | Vérifie reproductibilité du générateur et séparation des populations train/évaluation. | Test / données |
| `tests/explainability/test_shap_service.py` | Teste additivité, schéma, catégories mixtes, TreeExplainer, regroupement one-hot, importances et PNG locaux/globaux. | Test / ML |
| `tests/gating/test_gate.py` | Teste PASS seulement si F1 et AUC dépassent les marges, FAIL conserve alias, et valide hash/trigger/schémas. | Test / MLOps |
| `tests/monitoring/test_psi.py` | Teste profils stable/drift, catégories nouvelles, manquants, fenêtres et configurations PSI invalides. | Test / monitoring |
| `tests/training/__init__.py` | Marque le sous-package tests d'entraînement. | Test / support |
| `tests/training/test_tracking.py` | Teste run MLflow, paramètres, métriques, signature/artefacts de `train_and_log` sur données temporaires. | Test / MLOps |

### Documentation, client et remise

| Fichier | Rôle, fonctionnement, interactions | Importance |
|---|---|---|
| `docs/dataset-readiness.md` | État et procédure de reprise du dataset officiel, contrôles preflight et décisions attendues. | Support / gouvernance |
| `docs/dataset-validation-2026-11-21.md` | Résultats datés de validation, populations, contrôles et hashes du dataset synthétique autorisé. | Support / preuve |
| `docs/phase-1.md` | Démarche, commandes, livrables, rôles et documentation des lots WP1–WP8. | Support / guide |
| `docs/remaining-delivery-tasks.md` | Statut des neuf tâches, périmètre synthétique approuvé, preuves, limites et livrables restants hors acceptation technique. | Support / pilotage |
| `docs/revue-conformite-wp0-wp8.md` | Matrice d'alignement avec l'énoncé/cahier, architecture, écarts, exigences et feuille de route WP0–WP8. | Support / audit |
| `docs/synthetic-demo.md` | Journal chronologique des runs, versions, requêtes, valeurs SHAP/PSI, rapports gate et limites synthétiques. | Support / preuve |
| `docs/wp1-wp4-verification.md` | Résumé de vérification des fondations data et du serving/API initial. | Support / preuve |
| `docs/wp5-shap.md` | Guide d'exploitation SHAP : méthodes, références, contrats HTTP, coûts et précautions d'interprétation. | Support / guide |
| `docs/wp6-psi.md` | Guide PSI : calcul, configuration, endpoints/CLI, rapports et limites de seuil. | Support / guide |
| `docs/wp7-gate.md` | Guide challenger, critères PASS/FAIL, scripts, alias, rapports et tests de promotion. | Support / guide |
| `docs/wp8-delivery.md` | Guide Compose, services, commandes, Postman, tests et preuves de livraison/limites. | Support / guide |
| `docs/codebase-file-report.md` | Rapport de cet audit : rôle, fonctionnalités, dépendances, importance et décisions de nettoyage pour les fichiers accessibles. | Support / audit |
| `postman/Project35_WP1-WP7.postman_collection.json` | Client HTTP pour health, contrat, scoring, OpenAPI, SHAP et PSI; variables de base et payloads synthétiques. Consomme FastAPI/Django/MLflow. | Support / test manuel |
| `presentation/outline-wp8.md` | Plan complet de 20 diapositives indicatives, preuves disponibles et limites synthétiques; consommé pour la démonstration. | Support / présentation |
| `deliverables/Project35-synthetic-technical-demo-2026-11-21.zip` | Paquet reproductible de documentation, sources, tests et rapports synthétiques; inventaire vérifié sans CSV bruts, DB, modèle sérialisé ou `.env`. | Livrable |

### Rapports d'exécution (`reports/`)

| Fichier | Rôle, fonctionnement, interactions | Importance |
|---|---|---|
| `reports/synthetic/phase1-data-quality.json` | Résultat WP1 : readiness, qualité, tailles de splits et features; sorti par preflight/phase1. | Support / preuve |
| `reports/synthetic/training-run.json` | Résumé non sensible du run LR, modèle URI, métriques, hash et avertissements; renvoie au run MLflow. | Support / preuve |
| `reports/synthetic/psi-stable.json` | Rapport d'une fenêtre stable : référence/courant, hashes, seuils, status et détail feature. | Support / preuve |
| `reports/synthetic/psi-drift.json` | Rapport d'une fenêtre décalée artificiellement; déclenchement de l'alerte de démonstration. | Support / preuve |
| `reports/synthetic/wp7-gate-fail.json` | Résultat FAIL initial, champion/challenger, hashes, métriques, critères et absence de promotion. | Support / preuve MLOps |
| `reports/synthetic-gates/1ae29d2e-674f-40a3-b804-b9f65e333191.json` | Rapport d'un second gate FAIL avec identifiant différent; preuve d'une exécution distincte, non copie du rapport initial. | Support / preuve MLOps |
| `reports/synthetic/wp8-smoke.json` | Résumé synthétique des statuts d'API, modèle, SHAP, PSI, tests et limites. | Support / preuve livraison |

## Éléments retirés pendant cette intervention

Les chemins suivants sont des caches régénérables supprimés le 2026-11-21. Aucun fichier source correspondant n'a été supprimé.

**Fichier couverture :** `.coverage`.

**Cache pytest :** `.pytest_cache/.gitignore`, `.pytest_cache/CACHEDIR.TAG`, `.pytest_cache/README.md`, `.pytest_cache/v/cache/lastfailed`, `.pytest_cache/v/cache/nodeids`.

**Bytecode Python (29 fichiers) :**

- `django_app/config/__pycache__/{settings,urls,wsgi,__init__}.cpython-311.pyc`
- `django_app/predictions/__pycache__/{apps,explainability,explain_serializers,monitoring,monitoring_serializers,serializers,services,views,__init__}.cpython-311.pyc`
- `src/project35/__pycache__/__init__.cpython-311.pyc`
- `src/project35/api/__pycache__/{main,__init__}.cpython-311.pyc`
- `src/project35/data/__pycache__/{contract,phase1,__init__}.cpython-311.pyc`
- `src/project35/explainability/__pycache__/{shap_service,__init__}.cpython-311.pyc`
- `src/project35/gating/__pycache__/{service,__init__}.cpython-311.pyc`
- `src/project35/monitoring/__pycache__/{psi,__init__}.cpython-311.pyc`
- `src/project35/registry/__pycache__/{service,__init__}.cpython-311.pyc`
- `src/project35/training/__pycache__/{trainer,__init__}.cpython-311.pyc`

## Suppressions préexistantes, non effectuées par cet audit

Avant cette intervention, Git signalait déjà **66 fichiers supprimés du working tree** sous `.tmp_pdf_lib/`. Ils constituent une copie extraite de `pypdf 7.1.1`; le dossier parent vide/inaccessible est toujours présent. L'audit n'a ni provoqué ces suppressions, ni restauré, ni stage, ni nettoyé ces chemins. Aucune référence à `tmp_pdf_lib`/`pypdf` n'a été trouvée dans le code/configuration/documentation accessibles. Liste exacte des suppressions observées :

```text
.tmp_pdf_lib/pypdf-7.1.1.dist-info/INSTALLER
.tmp_pdf_lib/pypdf-7.1.1.dist-info/METADATA
.tmp_pdf_lib/pypdf-7.1.1.dist-info/RECORD
.tmp_pdf_lib/pypdf-7.1.1.dist-info/REQUESTED
.tmp_pdf_lib/pypdf-7.1.1.dist-info/WHEEL
.tmp_pdf_lib/pypdf-7.1.1.dist-info/licenses/LICENSE
.tmp_pdf_lib/pypdf/__init__.py
.tmp_pdf_lib/pypdf/_cmap.py
.tmp_pdf_lib/pypdf/_codecs/__init__.py
.tmp_pdf_lib/pypdf/_codecs/_codecs.py
.tmp_pdf_lib/pypdf/_codecs/adobe_glyphs.py
.tmp_pdf_lib/pypdf/_codecs/core_font_metrics.py
.tmp_pdf_lib/pypdf/_codecs/pdfdoc.py
.tmp_pdf_lib/pypdf/_codecs/std.py
.tmp_pdf_lib/pypdf/_codecs/symbol.py
.tmp_pdf_lib/pypdf/_codecs/zapfding.py
.tmp_pdf_lib/pypdf/_configuration.py
.tmp_pdf_lib/pypdf/_crypt_providers/__init__.py
.tmp_pdf_lib/pypdf/_crypt_providers/_base.py
.tmp_pdf_lib/pypdf/_crypt_providers/_cryptography.py
.tmp_pdf_lib/pypdf/_crypt_providers/_fallback.py
.tmp_pdf_lib/pypdf/_crypt_providers/_pycryptodome.py
.tmp_pdf_lib/pypdf/_doc_common.py
.tmp_pdf_lib/pypdf/_encryption.py
.tmp_pdf_lib/pypdf/_page.py
.tmp_pdf_lib/pypdf/_page_labels.py
.tmp_pdf_lib/pypdf/_protocols.py
.tmp_pdf_lib/pypdf/_reader.py
.tmp_pdf_lib/pypdf/_text_extraction/__init__.py
.tmp_pdf_lib/pypdf/_text_extraction/_layout_mode/__init__.py
.tmp_pdf_lib/pypdf/_text_extraction/_layout_mode/_fixed_width_page.py
.tmp_pdf_lib/pypdf/_text_extraction/_layout_mode/_text_state_manager.py
.tmp_pdf_lib/pypdf/_text_extraction/_layout_mode/_text_state_params.py
.tmp_pdf_lib/pypdf/_text_extraction/_text_extractor.py
.tmp_pdf_lib/pypdf/_utils.py
.tmp_pdf_lib/pypdf/_version.py
.tmp_pdf_lib/pypdf/_writer.py
.tmp_pdf_lib/pypdf/actions/__init__.py
.tmp_pdf_lib/pypdf/actions/_actions.py
.tmp_pdf_lib/pypdf/annotations/__init__.py
.tmp_pdf_lib/pypdf/annotations/_base.py
.tmp_pdf_lib/pypdf/annotations/_markup_annotations.py
.tmp_pdf_lib/pypdf/annotations/_non_markup_annotations.py
.tmp_pdf_lib/pypdf/constants.py
.tmp_pdf_lib/pypdf/errors.py
.tmp_pdf_lib/pypdf/filters.py
.tmp_pdf_lib/pypdf/generic/__init__.py
.tmp_pdf_lib/pypdf/generic/_appearance_stream.py
.tmp_pdf_lib/pypdf/generic/_base.py
.tmp_pdf_lib/pypdf/generic/_color.py
.tmp_pdf_lib/pypdf/generic/_data_structures.py
.tmp_pdf_lib/pypdf/generic/_files.py
.tmp_pdf_lib/pypdf/generic/_fit.py
.tmp_pdf_lib/pypdf/generic/_font.py
.tmp_pdf_lib/pypdf/generic/_image_inline.py
.tmp_pdf_lib/pypdf/generic/_image_xobject.py
.tmp_pdf_lib/pypdf/generic/_link.py
.tmp_pdf_lib/pypdf/generic/_outline.py
.tmp_pdf_lib/pypdf/generic/_rectangle.py
.tmp_pdf_lib/pypdf/generic/_utils.py
.tmp_pdf_lib/pypdf/generic/_viewerpref.py
.tmp_pdf_lib/pypdf/pagerange.py
.tmp_pdf_lib/pypdf/papersizes.py
.tmp_pdf_lib/pypdf/py.typed
.tmp_pdf_lib/pypdf/types.py
.tmp_pdf_lib/pypdf/xmp.py
```

## Contrôles et limites

Le nettoyage n'a changé aucun code/configuration; les 50 tests passés et 77% de couverture sont le résultat enregistré avant le nettoyage. Aucun test n'a été relancé afin de ne pas recréer les caches supprimés. Le statut Git après intervention reste les 66 suppressions préexistantes sous `.tmp_pdf_lib/`; aucun autre changement de code n'a été introduit par le nettoyage. Aucun fichier conservé n'est mis en quarantaine.
