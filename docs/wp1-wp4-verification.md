# Vérification des WP1 à WP4

Date de contrôle: 2026-11-19

## Résultats par lot

| Lot | Vérification | Résultat | Limite restante |
|---|---|---|---|
| WP1 — Données et pipeline | Contrat, profil de features, distributions de cible, split, prétraitement et baseline testés | Code validé | Dataset, cible et classe positive officiels absents; pas de rapport EDA réel ni d'audit métier de fuite |
| WP2 — MLflow Tracking | Run local SQLite, métriques validation, empreinte SHA-256, artefacts, signature et exemple vérifiés | Test synthétique réussi | Aucun run sur le dataset officiel |
| WP3 — Model Registry | Enregistrement de versions, tags source-run, aliases et garde de promotion vérifiés | Test synthétique réussi | Le champion réel reste à initialiser après évaluation de vraies données |
| WP4 — Django REST | `/health/`, `/api/predict/`, `/api/model-info/`, OpenAPI/Swagger, validation, 503 et chargement d'alias vérifiés | Tests HTTP et test registry→API réussis | Accès local anonyme seulement; aucune image Docker n'a été démarrée |

## Contrôles exécutés

- `pipenv run pytest --cov=project35 --cov-report=term-missing`: 19 tests réussis.
- `pipenv run python django_app/manage.py check`: aucun problème Django.
- `pipenv run python django_app/manage.py migrate --check --noinput`: aucune migration en attente.
- `docker compose config --quiet`: configuration Compose valide.
- Test ciblé registry→Django REST: run MLflow synthétique enregistré, alias champion résolu et prédiction/probabilités retournées par l'API.

Deux avertissements de dépendances apparaissent sans faire échouer les tests: dépréciation de l'adaptateur Starlette/httpx et dépréciation SQLAlchemy interne au backend MLflow.

## Conditions avant validation métier

1. Renseigner `configs/data_contract.yaml` avec le dataset, la cible, la classe positive et le mode F1 imposés.
2. Exécuter le profil EDA et examiner les features avec l'équipe pour détecter fuite de cible, variables post-événement, IDs et colonnes métier incorrectes.
3. Entraîner puis enregistrer le vrai modèle; initialiser `@champion` seulement après validation des métriques sur l'échantillon retenu.
4. Ajouter authentification et contrôle d'accès avant toute exposition hors machine locale.
5. Construire et lancer Compose dans l'environnement cible avant la démonstration.

Ce rapport est un instantané daté du 2026-11-19, avant l'implémentation WP5. Pour l'état consolidé des WP1–WP8, lire `docs/revue-conformite-wp0-wp8.md`; le WP5 est depuis implémenté et couvert par tests synthétiques.
