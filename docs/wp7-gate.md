# WP7 — Retraining, challenger et gate de promotion

## Objectif et statut

WP7 ajoute un workflow contrôlé qui entraîne un candidat, l'enregistre sous `@challenger`, mesure champion et challenger sur le même holdout étiqueté, puis ne déplace `@champion` que si les deux critères configurés passent. Le code et les tests synthétiques sont implémentés; aucun résultat de promotion sur les données du sujet n'est disponible, car le dataset et ses décisions métier restent à confirmer.

## Règles du gate

- Le holdout doit être indépendant du train, de la validation et du test final; il doit contenir les mêmes features et la cible contractuelle.
- Les modèles doivent exposer les mêmes noms et l'ordre des features, les mêmes classes dans le même ordre et `predict_proba` valide.
- F1 utilise `f1_average` et `positive_class` du contrat. ROC-AUC binaire suit la classe positive configurée; le cas multiclasse utilise OVR pondéré.
- Les seuils `min_f1_improvement` et `min_roc_auc_improvement` sont définis avant l'évaluation. La règle est stricte: delta > seuil pour les deux métriques. Une égalité ou un échec conserve le champion.
- Le hash SHA-256 du holdout est obligatoire. Le rapport indique versions, run IDs, hash, nombre de lignes, mesures, deltas, critères, déclencheur, horodatage et décision.
- Le rapport est enregistré comme `promotion_gates/<uuid>.json` dans le run challenger et sous `reports/gates/`. Le registre relit et vérifie le rapport du run/version concernés avant de déplacer l'alias.
- Le workflow peut être lancé manuellement ou avec un rapport PSI `status=drift`, `alert=true` et `report_id`. Le PSI déclenche l'entraînement; il n'autorise jamais à lui seul la promotion.

## Fichiers et responsabilités

| Fichier | Rôle |
|---|---|
| `src/project35/gating/service.py` | Vérifie le schéma et les probabilités; calcule F1/AUC, delta et décision; persiste le rapport MLflow. |
| `src/project35/registry/service.py` | Vérifie l'UUID et l'artefact de gate, ses versions, son hash, ses critères et ses deltas avant changement de `@champion`. |
| `src/project35/training/trainer.py` | Entraîne le pipeline demandé et trace son estimateur, paramètres, métriques et rapport dans MLflow. |
| `src/project35/data/phase1.py` | Construit le prétraitement commun et les classifieurs configurables. |
| `scripts/run_challenger_workflow.py` | Orchestre entraînement → registre challenger → comparaison → rapport → promotion/refus. |
| `scripts/train.py` | Permet d'exécuter séparément un run candidat via `--classifier` et `--classifier-params`. |
| `configs/gate.example.yaml` | Exemple des estimateurs, hyperparamètres, holdout, marges et dossier de rapports. |
| `tests/gating/test_gate.py`, `tests/training/test_tracking.py` | Cas synthétiques de pass/fail, gouvernance d'alias et trajet registre vers scoring Django. |

Les classifieurs candidats actuellement pris en charge sont `logistic_regression`, `random_forest`, `extra_trees` et `decision_tree`. Les transformations restent dans le même pipeline sklearn, et les paramètres effectifs sont tracés pour rendre le run reproductible.

## Mise en route

Préparer les configurations, sans ajouter de données confidentielles au dépôt:

```powershell
Copy-Item configs/data_contract.example.yaml configs/data_contract.yaml
Copy-Item configs/data_provenance.example.yaml configs/data_provenance.yaml
Copy-Item configs/mlflow.example.yaml configs/mlflow.yaml
Copy-Item configs/gate.example.yaml configs/gate.yaml
```

Renseigner dataset, cible, colonnes/split et provenance; exécuter `scripts/preflight_data.py --require-approved` avant entraînement. Configurer aussi serveur MLflow, modèle, holdout évalué et marges. Le holdout gate doit être séparé du CSV d'entraînement et avoir les mêmes features, plus la cible. Fixer/documenter les marges avant consultation des résultats. Le modèle champion doit déjà exister; pour la première version seulement, initialiser `@champion` avec `scripts/register_model.py`.

Lancer une comparaison manuelle:

```powershell
pipenv run python scripts/run_challenger_workflow.py `
  --data-config configs/data_contract.yaml `
  --provenance configs/data_provenance.yaml `
  --gate-config configs/gate.yaml `
  --mlflow-config configs/mlflow.yaml
```

Lancer le même workflow à partir d'un rapport de dérive validé:

```powershell
pipenv run python scripts/run_challenger_workflow.py `
  --data-config configs/data_contract.yaml `
  --provenance configs/data_provenance.yaml `
  --gate-config configs/gate.yaml `
  --mlflow-config configs/mlflow.yaml `
  --psi-report reports/monitoring/psi_report.json
```

Le processus retourne 0 sur PASS et 1 sur FAIL. FAIL est une décision attendue: le rapport est conservé et l'alias champion reste inchangé. Toute erreur de configuration ou d'exécution arrête le workflow avec un message CLI.

## Smoke test synthétique WP7

Le 2026-11-20, `reports/synthetic/wp7-gate-fail.json` a été produit en déclenchant le workflow avec un rapport PSI `drift` synthétique. Le run candidat Random Forest (`4b6c2d0e8cae45a1860274e8160ef22c`) a été enregistré comme version 2 sous le modèle isolé `project35-synthetic-churn`. Sur le holdout synthétique de 1 000 lignes, F1 augmente de 0.2660, mais ROC-AUC diminue de 0.0161; le gate a donc refusé la promotion. L'alias `@champion` reste en version 1 et `@challenger` pointe vers la version 2. Cette preuve valide la branche FAIL et n'est pas une décision métier.

Pour le modèle arbre entraîné par le pipeline, `trainer.py` déclare `sklearn.tree._tree.Tree` comme seul type skops supplémentaire de confiance et seulement lorsque l'estimateur est DecisionTree, RandomForest ou ExtraTrees. Les autres types restent soumis au contrôle skops.

## Tests et limites

```powershell
pipenv run pytest tests/gating tests/training/test_tracking.py tests/data/test_phase1.py -q
```

Les tests utilisent des estimateurs et des données synthétiques temporaires; ils valident la logique et la gouvernance, pas la qualité métier, les seuils, les licences, ni une promotion réelle. Les dépendances (MLflow, serveur/artifact store et dataset approuvé) doivent être accessibles pour un run de bout en bout. Le déclenchement PSI est une commande ponctuelle, pas un ordonnanceur/background worker. WP8 a validé le profil technique synthétique sous Compose; la remise métier exige encore dataset approuvé, marges définies et preuves de cours.
