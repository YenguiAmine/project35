# Suivi des neuf tâches de livraison — 2026-11-21

## Décision de périmètre approuvée

Le 2026-11-21, l'utilisateur a explicitement autorisé l'emploi du dataset synthétique déjà présent (`data/synthetic/`) ainsi que de son contrat de données et de sa politique d'évaluation, exclusivement dans leur périmètre synthétique. Cette décision est consignée dans `configs/data_provenance.synthetic.yaml`, `configs/data_contract.synthetic.yaml` et `configs/gate.synthetic.yaml`.

Cette approbation n'est ni une approbation de l'enseignant/data owner ni une licence d'un dataset réel. Les labels sont générés, les métriques illustrent le fonctionnement logiciel et aucun résultat ne doit être présenté comme preuve métier ou comme résultat officiel du cours.

## Tableau de statut

| # | Tâche | Statut synthétique au 2026-11-21 | Preuve / restant |
|---:|---|---|---|
| 1 | Source et autorisation d'usage | **Terminée pour le dataset synthétique uniquement** | Manifeste + provenance et SHA-256 dans `data/synthetic/manifest.json` et `configs/data_provenance.synthetic.yaml`. L'obtention d'un jeu officiel reste hors du périmètre autorisé. |
| 2 | Contrat et politique d'évaluation | **Approuvés pour le périmètre synthétique** | Contrat stratifié, cible `churn`, classe positive 1, seed 35; gate RF et marges delta strictement positives documentés sous `configs/*synthetic.yaml`. Ces marges ne sont pas des seuils métier. |
| 3 | Qualité, fuite et baseline | **Terminée** | Preflight `ready_for_review`, zéro valeur manquante/doublon, splits 2400/800/800, rapport `docs/dataset-validation-2026-11-21.md`, hash dans le manifeste; WP1–WP2 consignés dans `docs/synthetic-demo.md`. |
| 4 | Comparaison et sélection de modèles | **Terminée pour le périmètre synthétique** | Quatre candidats par défaut comparés sur le même split de validation; mêmes hashes et seed. Aucun ne bat LR sur les deux marges approuvées: le champion synthétique est conservé. Runs et décision dans `docs/synthetic-demo.md`. |
| 5 | Registry et API | **Terminée pour la démonstration synthétique** | Modèle isolé `project35-synthetic-wp1-5-20261121@champion` v1, Django temporaire, model-info/predict/health/Swagger HTTP 200; modèle métier non modifié. Preuves dans `docs/synthetic-demo.md` et `reports/synthetic/wp8-smoke.json`. |
| 6 | SHAP et références de monitoring | **TreeExplainer livré; validation PSI synthétique terminée** | Le service choisit TreeExplainer pour les pipelines DecisionTree/RandomForest/ExtraTrees, regroupe les contributions one-hot par feature source et conserve Permutation SHAP en fallback. PSI stable/drift et hashes consignés. Les seuils PSI restent illustratifs. |
| 7 | Gate WP7 | **Terminée pour la démonstration synthétique** | Deux décisions FAIL reproductibles, champion conservé v1, challenger v3; hash holdout et rapports distincts dans `reports/synthetic-gates/` et `docs/synthetic-demo.md`. Aucun PASS n'a promu de modèle sur données métier; la branche PASS est couverte par les tests antérieurement consignés. |
| 8 | Démonstration et preuves techniques | **Terminée et suite rejouée** | Suite après intégration API TreeExplainer, dans le conteneur Python 3.11 avec code/test actuels montés en lecture seule : **50 passed, 77% couverture, 63,66 s, 5 avertissements**; Django check propre, Compose config et health/Swagger HTTP 200. Détails dans `docs/wp8-delivery.md`. |
| 9 | Présentation et archive | **Terminée pour la livraison technique synthétique** | Le plan existant de 20 diapositives est renseigné avec les preuves du périmètre synthétique. L'archive `deliverables/Project35-synthetic-technical-demo-2026-11-21.zip` contient 84 chemins uniques sans CSV/DB/modèle/secret. Une adaptation de forme peut être nécessaire si le cours impose un export différent; cette archive ne prétend pas être une soumission métier officielle. |

## Étapes restantes, dans l'ordre

1. Confirmer avec l'enseignant la forme exacte d'affichage/acceptation SHAP demandée; le calcul TreeExplainer est maintenant implémenté et testé. Aucune explication n'est causale.
2. Si une soumission officielle est ensuite demandée, appliquer son format institutionnel au plan existant, après réception des règles du cours.

## Compte honnête

Les **9 tâches de livraison technique sont complétées dans le périmètre synthétique approuvé**. La comparaison utilise les quatre candidats, l'explicabilité couvre TreeExplainer et Permutation, la suite a été rejouée, et le plan/archive sont vérifiés. Les acceptations externes (dataset métier/cours, règles de présentation institutionnelle ou validation du data owner) restent hors du périmètre et ne sont pas présentées comme des tâches techniques achevées ni comme approbations déjà reçues.

## Références existantes

- Validation et empreintes : [dataset-validation-2026-11-21.md](dataset-validation-2026-11-21.md), `data/synthetic/manifest.json`.
- Journal des exécutions : [synthetic-demo.md](synthetic-demo.md).
- WP5 : [wp5-shap.md](wp5-shap.md). WP6 : [wp6-psi.md](wp6-psi.md). WP7 : [wp7-gate.md](wp7-gate.md). WP8 : [wp8-delivery.md](wp8-delivery.md).
- Plan de présentation existant : [outline-wp8.md](../presentation/outline-wp8.md).