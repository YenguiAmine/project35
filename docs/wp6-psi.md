# WP6 — Monitoring de dérive PSI

## Objectif

Mesurer les changements de distribution entre une population de référence stable et un batch courant, sans utiliser le label courant. Le PSI signale une dérive de covariables; il ne prouve pas une baisse de performance et ne lance pas à lui seul la promotion d’un modèle.

## Calcul et conventions

Le profile est ajusté une seule fois sur la référence et réutilisé pour chaque fenêtre:

- Numérique: quantiles de référence pour les coupures internes; les mêmes coupures sont conservées pour toutes les fenêtres. Les valeurs sous le minimum/au-dessus du maximum de référence vont dans les bins underflow/overflow; les valeurs manquantes sont un bin séparé.
- Catégoriel: catégories de référence figées; toute catégorie nouvelle devient special:other; les valeurs manquantes restent special:missing.
- Chaque proportion utilise un epsilon positif pour que le logarithme soit défini. Le calcul par bin est (Q_i - P_i) * ln(Q_i / P_i) et le score de feature est la somme; P est référence et Q courant.
- Les seuils sont configurables: status stable si PSI < watch; watch si watch <= PSI <= drift; drift si PSI > drift. Defaults indicatifs: watch=0.10, drift=0.25. Ces valeurs ne remplacent pas les seuils que l’enseignant pourrait imposer.
- Le status global est la sévérité maximale des features. L’API renvoie uniquement distributions agrégées, pas les lignes brutes.
- Les rapports incluent les définitions de bins, fenêtre, date, nombre de lignes, seuils, proportions et SHA-256 référence/courant pour audit.

## Configuration

Fournir un CSV de référence qui contient exactement les features à surveiller, avec types cohérents, sans cible/ID inutiles. Utiliser train approuvé ou une fenêtre stable et figer le fichier; ne pas ajuster les coupures sur le batch courant.

Variables d’environnement:

- PROJECT35_PSI_REFERENCE: chemin du CSV de référence.
- PSI_BINS: nombre de bins quantiles numériques, défaut 10.
- PSI_EPSILON: smoothing strictement positif, défaut 0.000001.
- PSI_WATCH_THRESHOLD: seuil watch, défaut 0.10.
- PSI_DRIFT_THRESHOLD: seuil drift, défaut 0.25.

Compose monte data en lecture seule et pointe par défaut vers /app/data/processed/psi_reference.csv. En local:

```powershell
$env:PROJECT35_PSI_REFERENCE = "data/processed/psi_reference.csv"
$env:PSI_BINS = "10"
$env:PSI_WATCH_THRESHOLD = "0.10"
$env:PSI_DRIFT_THRESHOLD = "0.25"
```

## API Django REST

Swagger: http://127.0.0.1:8001/api/docs/.

POST /api/monitoring/psi/

```json
{
  "current_window": "2026-11-20-batch-001",
  "records": [
    {"feature_num_1": 12.4, "feature_cat_1": "A"},
    {"feature_num_1": 22.1, "feature_cat_1": "NEW"}
  ]
}
```

records est limité à 10 000 entrées par requête. Les champs doivent avoir un schéma non vide identique entre lignes et correspondre exactement aux features de référence. Réponse: report_id, hashes, fenêtre/date, paramètres, status/alert et détails par feature, incluant PSI et bins.

Codes attendus: 200 calcul terminé; 400 payload/schema courant invalide; 503 référence/configuration indisponible. Une erreur PSI n’enregistre pas les records; les logs ne contiennent que identifiant, taille, status, nb features et hashes.

## CLI batch

Préparer un fichier courant avec les mêmes colonnes et types, puis lancer:

```powershell
pipenv run python scripts/monitor_psi.py --reference data/processed/psi_reference.csv --current data/current/batch.csv --window "2026-11-20-batch-001" --output reports/monitoring/psi_report.json
```

La commande produit un rapport JSON partageable et retourne une erreur explicite si les fichiers/schémas ne sont pas valides.

## Code et preuves

- src/project35/monitoring/psi.py: profile figé, calcul des bins/proportions/contributions et status.
- django_app/predictions/monitoring.py: chargement/cache du profile par chemin/date/taille; les changements de fichier invalident le cache.
- django_app/predictions/monitoring_serializers.py, views.py, config/urls.py: contrat REST, validation, journal de rapport et OpenAPI.
- scripts/monitor_psi.py: traitement batch.
- tests/monitoring/test_psi.py, tests/api/test_django_scoring_api.py: cas identiques, dérive, valeurs manquantes, catégories inédites, schéma et API.
- `postman/Project35_WP1-WP7.postman_collection.json`: requête PSI avec un payload conforme au profil synthétique Compose; ajuster les features pour une autre référence.

## Limites et critères de sortie

Le PSI surveille la dérive feature sans label; dès que les labels sont disponibles, compléter par métriques de performance et analyse de biais. Ne pas déclencher automatiquement retraining/promote avant WP7: WP7 devra utiliser le PSI comme un signal, puis comparer challenger/champion sur une population indépendante selon des seuils préconfigurés.

La validation pytest actuelle utilise seulement des données synthétiques. Le dataset réel, la référence approuvée et les seuils métier restent à fournir/valider.

