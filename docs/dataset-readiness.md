# Préparation et approbation des données

## État observé

Le répertoire `data/` ne contient aucun dataset métier. Les PDF du projet décrivent la plateforme et ses lots, mais ne définissent pas à eux seuls le dataset autorisé, les colonnes, la cible, la classe positive ni les seuils du gate. Aucun de ces éléments n'est inventé ici.

Pour permettre une démonstration technique pendant l'attente, le dépôt contient maintenant une population **entièrement synthétique** B2B churn: 4 000 lignes d'apprentissage et un holdout gate indépendant de 1 000 lignes. Voir [data/synthetic/README.md](../data/synthetic/README.md), le générateur et les configurations `*.synthetic.yaml`. Cette population n'est pas le dataset officiel et ses mesures ne sont pas des résultats métier.

La recréer avec la graine par défaut:

```powershell
pipenv run python scripts/generate_synthetic_dataset.py
pipenv run python scripts/preflight_data.py --config configs/data_contract.synthetic.yaml --provenance configs/data_provenance.synthetic.yaml --require-approved
```

Le statut `approved` dans la fiche synthétique autorise uniquement les outils locaux sur ces données générées; il ne vaut pas approbation du dataset, des variables ou des seuils du projet de cours.

Parcours d'entraînement et de gate synthétique, après démarrage de Compose et MLflow:

```powershell
pipenv run python scripts/phase1.py --config configs/data_contract.synthetic.yaml --provenance configs/data_provenance.synthetic.yaml
pipenv run python scripts/train.py --data-config configs/data_contract.synthetic.yaml --provenance configs/data_provenance.synthetic.yaml --mlflow-config configs/mlflow.synthetic.yaml
pipenv run python scripts/register_model.py --run-id <RUN_ID> --model-name project35-synthetic-churn --alias champion --mlflow-config configs/mlflow.synthetic.yaml
pipenv run python scripts/run_challenger_workflow.py --data-config configs/data_contract.synthetic.yaml --provenance configs/data_provenance.synthetic.yaml --gate-config configs/gate.synthetic.yaml --mlflow-config configs/mlflow.synthetic.yaml
```

Le gate peut refuser de promouvoir le challenger; cette issue est un résultat valide du test de workflow. Aucun seuil n'est calibré pour le churn réel.

## Étapes pour reprendre avec le dataset approuvé

1. Obtenir l'accord de l'enseignant/data owner sur la source, l'usage, la licence, le cas retenu et la cible.
2. Copier `configs/data_contract.example.yaml` vers `configs/data_contract.yaml` et `configs/data_provenance.example.yaml` vers `configs/data_provenance.yaml`.
3. Placer le CSV original approuvé dans `data/raw/`; renseigner source/version/date/licence, classe positive, politique anti-fuite, revue des champs sensibles, et état `pending` dans la provenance.
4. Exécuter une pré-validation sans ajuster de modèle:

   ```powershell
   pipenv run python scripts/preflight_data.py `
     --config configs/data_contract.yaml `
     --provenance configs/data_provenance.yaml
   ```

5. Revoir la qualité, les doublons, les valeurs manquantes, la distribution cible, les features après exclusions et les tailles/distributions de train-validation-test. Résoudre les anomalies et décider la stratégie temporelle/entity-aware si nécessaire.
6. Faire approuver le contrat, les définitions de classe, les exclusions, le split, la licence et les seuils par les responsables. Reporter le `dataset_sha256` produit dans la provenance et définir `approval_status: approved`.
7. Réexécuter en mode bloquant:

   ```powershell
   pipenv run python scripts/preflight_data.py `
     --config configs/data_contract.yaml `
     --provenance configs/data_provenance.yaml `
     --require-approved
   ```

Le rapport est écrit dans `reports/preflight/data_readiness.json`. Il contient des statistiques de qualité et métadonnées, pas les lignes brutes. Les CSV et configurations locales sont ignorés par Git.

Les commandes d'entraînement `scripts/phase1.py`, `scripts/train.py` et `scripts/run_challenger_workflow.py` exigent cette provenance approuvée et vérifient à nouveau le hash avant le fit. Le résumé non sensible de la provenance est rattaché au rapport d'évaluation du run MLflow.

## Ce que le contrôle vérifie

- Configuration renseignée, CSV lisible, colonnes obligatoires et cible sans valeurs nulles, plusieurs classes.
- Métadonnées provenance complètes, accord entre target/class positive du contrat et de la fiche, date/licence, revue des champs sensibles et correspondance SHA-256.
- Faisabilité des splits déclarés, présence de toutes les classes dans chaque split, tailles, comptages de classes et colonnes après retrait des IDs/dérivation de dates.
- En mode `--require-approved`, la source porte un état approuvé et un hash exact de la version inspectée.

Ce contrôle ne prouve pas l'absence de fuite sémantique, ne vérifie pas les droits auprès de l'éditeur, n'évalue pas l'équité et ne valide pas la représentativité. Ces décisions requièrent la revue métier. Le préflight est un livrable de préparation; poursuivre ensuite avec EDA approuvée, baseline/MLflow et le holdout de gate maintenu à part.
