# Démonstration synthétique WP1–WP7

## Décision d'usage — 2026-11-21

L'utilisateur a approuvé l'usage du dataset synthétique existant ainsi que son contrat de données et sa politique d'évaluation, uniquement pour l'implémentation et la démonstration technique. La décision est enregistrée dans `configs/data_provenance.synthetic.yaml` et `configs/*synthetic.yaml`. Elle n'approuve pas de dataset métier/cours, et ne transforme pas les scores ci-dessous en preuves métier. Les empreintes ont été revérifiées depuis les fichiers présents; aucun CSV de remplacement ni rapport redondant n'a été créé.

## Portée

Cette exécution vérifie le pipeline et le tracking. Le jeu a été généré par `scripts/generate_synthetic_dataset.py`: 4 000 observations d'entraînement et 1 000 holdout gate générés indépendamment. Les labels sont simulés; aucun client réel n'est représenté.

## Résultats consignés

- Hash train: `6167d68f99c93fb74c89f0cf3da6866d94d2bb1fc29723178e8f4249c49ebbec`.
- Split interne stratifié: 2 400 train, 800 validation, 800 test; les deux classes figurent dans chaque partition.
- Aucun doublon exact ni valeur manquante dans le CSV train.
- Run MLflow `c2a0c9ee8de04bf5a754c3e241f6b930`, modèle Logistic Regression; UI `http://127.0.0.1:5000/#/experiments/1/runs/c2a0c9ee8de04bf5a754c3e241f6b930`.
- Métriques de validation: F1 0.2673, precision 0.7714, recall 0.1617, accuracy 0.8150, ROC-AUC 0.7654, average precision 0.4833.
- Détails de l'exécution: `reports/synthetic/training-run.json` (fichier local ignoré par Git); le rapport WP1 est `reports/synthetic/phase1-data-quality.json`.

Ces métriques reflètent uniquement la règle générative synthétique; elles ne valident pas la qualité d'un système de churn et ne sont pas des résultats de l'énoncé.

## WP3 — Model Registry et chargement

Le run `c2a0c9ee8de04bf5a754c3e241f6b930` a été enregistré dans MLflow sous le nom isolé `project35-synthetic-churn`, version `1`, alias `champion`. La description de version précise que les labels et résultats sont synthétiques. Le nom métier `project35-classifier` n'a pas été modifié.

Smoke test réalisé depuis le conteneur API : chargement de `models:/project35-synthetic-churn@champion`, lecture de trois lignes sans la cible, prédictions `[0, 0, 0]`. La signature MLflow expose 12 caractéristiques requises : 8 numériques (dont `auto_renew` booléen) et 4 catégorielles. Les artefacts se chargent depuis le registre via le proxy MLflow.

L'enregistrement initial a échoué une fois avec `Run not found`; l'interrogation de l'expérience active a confirmé que le run existait et la seconde tentative a réussi. MLflow a averti que le chemin d'artefact historique `baseline_pipeline` était absent et a utilisé l'artefact modèle enregistré `models:/m-4bb7ca1aafc14169aae29fcfee370c2b`. Le chargement par alias et la prédiction ont confirmé que la version créée est exploitable.

## WP4 — API de scoring isolée

Une instance Django temporaire a été lancée sur `http://127.0.0.1:8002` avec `MLFLOW_REGISTERED_MODEL_NAME=project35-synthetic-churn`; le service habituel sur le port 8001 et son modèle configuré n'ont pas été modifiés. Le conteneur temporaire a été arrêté après les tests.

Tests HTTP effectués :

- `GET /api/model-info/` → 200, modèle `project35-synthetic-churn`, alias `champion`, version `1`, run et métriques retournés.
- `POST /api/predict/` avec une instance respectant les 12 champs MLflow → 200, `request_id` retourné, prédiction `[0]`, probabilités `[0.7954, 0.2046]`.
- `GET /health/` → `degraded` avant chargement à froid, puis `ok`, modèle chargé, version `1` après la première requête de scoring.
- `GET /api/schema/` et `GET /api/docs/` → 200. L'interface Swagger est disponible à `http://127.0.0.1:8002/api/docs/` pendant qu'une instance configurée est lancée.

Le warning OpenAPI `Status56fEnum` vient d'une collision de noms d'enum `status` dans des composants de schéma; le schéma est néanmoins servi en 200 et expose `/api/predict/`. Ce point est cosmétique et peut être renommé dans la configuration drf-spectacular si l'on veut un schéma sans warning.

## WP5 — Explicabilité SHAP

Un CSV de référence synthétique de 100 lignes, échantillonné depuis le train avec seed 35 et sans la cible, est disponible à `data/synthetic/b2b_churn_shap_reference.csv` (SHA-256 `3e8d131777822d62e8e09e3bc54c0ae6cf7d20805433e359bf244455ea8e32c4`). Il sert uniquement à la démonstration logicielle. Les limites de ce smoke test étaient 20 lignes de fond, 3 lignes globales et 25 évaluations.

Le premier appel a révélé que SHAP/NumPy ne supporte pas `isclose` sur les catégories texte du modèle. Le service encode les colonnes catégorielles uniquement dans l'espace de masquage numérique, puis les décode avant l'appel au pipeline sklearn; la requête et le modèle conservent les catégories d'origine. Un test de régression mixte numérique/catégoriel a été ajouté.

Tests HTTP sur l'image Django reconstruite, toujours isolée sur `8002` : `POST /api/explain/` → 200, 12 contributions, waterfall PNG décodable, probabilité `0.20460148` et résidu additif proche de `-2.8e-17`; `GET /api/explain/global/?class_index=1` → 200, 3 observations, 12 importances, PNG bar et beeswarm décodables. Les deux réponses portent modèle/version et hash du fond. Le test Pytest `tests/explainability/test_shap_service.py` passe : **4 passed**.

Pour la lecture du beeswarm, les couleurs des catégories représentent leurs codes entiers locaux à l'échantillon de référence, pas une grandeur ordinale métier. L'importance SHAP reste une attribution prédictive non causale. Le calcul utilise Permutation SHAP model-agnostic; réconcilier le TreeExplainer cité par l'énoncé reste un choix de conformité à confirmer avec un modèle arbre.

### Complément TreeExplainer — 2026-11-21

Le chemin TreeExplainer a ensuite été ajouté pour les pipelines sklearn DecisionTree, RandomForest et ExtraTrees. Les transformations sont reprises depuis le `ColumnTransformer` ajusté; les contributions one-hot sont regroupées sur les 12 features source pour les réponses locale et globale. La réponse expose `explainer_method`; Logistic Regression reste expliquée par PermutationExplainer. Un test service arbre mixte numérique/catégoriel vérifie la sortie additive et les trois PNG. Cela teste TreeExplainer sur pipeline d'arbre, mais le champion actuellement servi demeure LR; la démo HTTP existante utilise donc PermutationExplainer.

## WP6 — Monitoring de dérive PSI

Jeu de référence: `data/synthetic/b2b_churn_psi_reference.csv`, 4 000 lignes train sans cible, SHA-256 `ee87834969d7fe00f7902189e072df6278b7b9c3ff39c28fd13f5040b8485547`. Le batch stable vient des 1 000 lignes indépendantes du holdout gate, sans cible. Le batch drift est une copie synthétique du même holdout avec `product_usage_ratio_30d=0.99` et `region=synthetic_shift` sur toutes les lignes. Les trois CSV et leurs hashes sont décrits dans `data/synthetic/manifest.json`.

Résultats CLI (`scripts/monitor_psi.py`) et API (`POST /api/monitoring/psi/`) :

- Fenêtre `synthetic-stable`: status `stable`, alert `false`, 1 000 lignes; PSI usage `0.0061`, region `0.0053`. Rapport CLI `reports/synthetic/psi-stable.json`, report ID `a70ef6ad-8c5e-4640-82dd-51319b0158d3`; report ID HTTP `5f4bd68d-0a85-430c-ab4d-a2ca8d7d0206`.
- Fenêtre `synthetic-drift`: status `drift`, alert `true`, 1 000 lignes; PSI usage `18.6433`, region `41.7346`. Rapport CLI `reports/synthetic/psi-drift.json`, report ID `224da7cb-5d1a-4891-875b-b352d38299aa`; report ID HTTP `255d873e-6e7f-469f-8f42-6fc49459a26a`.

Une divergence de hash entre CLI et API a été détectée pendant la vérification : le CLI utilisait une sérialisation pandas, alors que l'API utilise le hash des octets du fichier. `scripts/monitor_psi.py` hache maintenant la référence brute comme l'API; les deux rapports CLI régénérés portent le même SHA-256 que l'API et le manifeste. Tests ciblés `tests/monitoring/test_psi.py` + `tests/api/test_django_scoring_api.py`: **19 passed**.

Ces scénarios prouvent les branches stable/alert et la traçabilité, pas une dérive réelle. La première alerte synthétique (`224da7cb-5d1a-4891-875b-b352d38299aa`) a déclenché WP7; la réexécution API ci-dessous n'a pas relancé le workflow.

Réexécution API le 2026-11-21 après WP1–WP5 : `wp6-rerun-stable` → `stable`, alert `false`, 1 000 lignes, PSI usage `0.0061`, region `0.0053`, report ID `e7633fcd-0d8b-4bf2-9231-195c0277289b`; `wp6-rerun-drift` → `drift`, alert `true`, 1 000 lignes, PSI usage `18.6433`, region `41.7346`, report ID `46e29b72-0ea6-41da-9285-9226d4b7adcc`. Les deux réponses portent le hash de référence `ee87834969d7fe00f7902189e072df6278b7b9c3ff39c28fd13f5040b8485547`. Cette réexécution n'a pas relancé le workflow WP7; l'alerte synthétique précédente, report ID `224da7cb-5d1a-4891-875b-b352d38299aa`, avait servi à le déclencher.

## WP7 — Retraining déclenché par PSI et gate challenger

Le rapport PSI `224da7cb-5d1a-4891-875b-b352d38299aa` (`synthetic-drift`, `drift`, `alert=true`) a déclenché le workflow. L'entraînement a produit le run `4b6c2d0e8cae45a1860274e8160ef22c`, enregistré comme `project35-synthetic-churn` version `2`, alias `challenger`. Le candidat Random Forest a été évalué contre le champion Logistic Regression version `1` sur le holdout gate indépendant (1 000 lignes; SHA-256 `7675b5d3e4e02c9c298af1bf630843118ea4cbbecb4e90a8278bd651598abda0`).

Décision du gate `8fc35192-6dd5-4d73-9a87-7bab9a8fb884`: **FAIL contrôlé**. F1 passe de `0.1860` à `0.4521` (delta `+0.2660`), tandis que ROC-AUC passe de `0.7412` à `0.7251` (delta `-0.0161`). La configuration synthétique exige des deltas strictement positifs pour les deux critères; le challenger n'est donc pas promu. La vérification MLflow confirme `@champion` reste en version `1` et `@challenger` pointe vers `2`. Le rapport JSON est dans `reports/synthetic/wp7-gate-fail.json`; MLflow contient aussi le run et l'artefact de gate.

Réexécution WP7 le 2026-11-21 avec le rapport PSI drift existant `224da7cb-5d1a-4891-875b-b352d38299aa`: run candidat `2f070c07c28e453b8ee657e8b38c2577`, version `3` sous `@challenger`; rapport distinct `reports/synthetic-gates/1ae29d2e-674f-40a3-b804-b9f65e333191.json`. Décision **FAIL**, F1 `0.4521` contre `0.1860`, ROC-AUC `0.7251` contre `0.7412`. L'alias `@champion` reste en version `1`; aucun rapport précédent n'a été écrasé. Cette seconde exécution reproduit le résultat attendu, sans constituer une validation métier.

Le premier run WP7 a exposé le contrôle de confiance `skops` avec RandomForest. L'entraînement n'autorisait que `numpy.dtype`; le sérialiseur exige aussi `sklearn.tree._tree.Tree`. `trainer.py` autorise désormais précisément ce type uniquement pour les classifieurs DecisionTree/RandomForest/ExtraTrees entraînés dans ce pipeline. Aucun chargement générique de types non approuvés n'est activé. L'exécution suivante a chargé et évalué le candidat correctement.

Les marges `0.0` et ces métriques sont des paramètres/résultats de démonstration synthétique, non des seuils ou preuves métier. Ce run valide le déclenchement PSI, la sérialisation, l'évaluation commune, le rapport et le refus de promotion; la branche PASS reste couverte par les tests unitaires, pas par une promotion sur les données officielles.

## Réexécution WP1–WP5 — 2026-11-21

Ces étapes ont été rejouées de bout en bout sur le profil synthétique; le modèle métier n'a pas été touché.

- **WP1:** préflight `ready_for_review`, zéro valeur manquante/doublon, splits 2 400/800/800. Le rapport généré est identique octet pour octet à `reports/synthetic/phase1-data-quality.json`; cette copie existante est réutilisée et aucun fichier de rapport en double n'est conservé.
- **WP2:** run MLflow `8713c1f94dea45608587336f8faf81e1`, Logistic Regression, dataset hash `6167d68f99c93fb74c89f0cf3da6866d94d2bb1fc29723178e8f4249c49ebbec`; F1 `0.2673`, precision `0.7714`, recall `0.1617`, accuracy `0.8150`, ROC-AUC `0.7654`, average precision `0.4833`. Les labels étant simulés, ce sont uniquement des métriques de démonstration. MLflow a averti sur la signature entière (si des valeurs manquantes étaient présentes à l'inférence) et l'absence d'exécutable Git pour les métadonnées de commit.
- **WP3:** modèle séparé `project35-synthetic-wp1-5-20261121`, version `1`, alias `champion`, lié à ce run. Le modèle précédent `project35-synthetic-churn` (champion v1/challenger v2) et le nom métier n'ont pas été modifiés.
- **WP4:** Django temporaire sur 8003, configuré sur ce nouveau modèle; `/api/model-info/` et `/api/predict/` réussissent. Prédiction `0`, probabilités `[0.5466, 0.4534]`, `request_id=eab9faad-9e67-48a9-b62f-5f7f950bde1b`; health `ok`, version `1` chargée.
- **WP5:** SHAP local/global réussit sur 12 features; le waterfall et les images bar/beeswarm sont des PNG décodables. Résidu d'additivité `-5.6e-17`; fond synthétique SHA-256 `3e8d131777822d62e8e09e3bc54c0ae6cf7d20805433e359bf244455ea8e32c4`. Le beeswarm encode les catégories de façon numérique locale et reste non causal.

L'instance temporaire 8003 a été arrêtée après les appels. Ces résultats rejouent les phases techniques, mais ne lèvent pas le blocage de validation métier faute de dataset officiellement approuvé.

## WP2 — Comparaison contrôlée des candidats — 2026-11-21

Quatre candidats ont été entraînés avec le même train (2 400 lignes), la même validation (800 lignes), le même prétraitement, la seed 35 et le même hash dataset `6167d68f99c93fb74c89f0cf3da6866d94d2bb1fc29723178e8f4249c49ebbec`. Les métriques ci-dessous sont exclusivement celles de validation. Les classifieurs utilisent les paramètres sklearn par défaut; `random_state=35` est fixé par le pipeline.

| Classifieur | Run MLflow | F1 validation | ROC-AUC validation | AP validation |
|---|---|---:|---:|---:|
| Logistic Regression | `8713c1f94dea45608587336f8faf81e1` | 0.2673 | **0.7654** | **0.4833** |
| Random Forest (défaut) | `fdf58ad7afc94f2c9e22fc94062f4bd6` | 0.1481 | 0.7406 | 0.4342 |
| Decision Tree | `6b6c969f5263421a809b5c01772d031f` | 0.2751 | 0.5379 | 0.2246 |
| Extra Trees | `855949c1c74c4a5980344afdf6238db7` | 0.2079 | 0.7245 | 0.4195 |

Décision conforme à `configs/gate.synthetic.yaml`: **conserver Logistic Regression comme champion synthétique**. Aucun des trois candidats par défaut n'améliore simultanément le F1 et l'ROC-AUC. Decision Tree gagne 0.0077 de F1 mais perd 0.2275 d'AUC; Random Forest et Extra Trees baissent les deux métriques par rapport à LR. Le RF optimisé du workflow WP7 (`2f070c07c28e453b8ee657e8b38c2577`, F1 val 0.4465, AUC val 0.7518) augmente F1 mais baisse aussi AUC; il reste un challenger soumis au gate, pas un candidat retenu. Aucune mesure du holdout final n'a servi à cette sélection. Ces runs terminent la comparaison technique prévue; ils ne produisent pas de conclusion métier.

## WP8 — Livraison technique et démonstration

Le profil Compose `synthetic` fournit un client HTTP relançable sur le port `8002`, connecté uniquement à `project35-synthetic-churn`, avec référence SHAP/PSI en lecture seule et volume Django synthétique dédié. Le service métier `django-api` reste sur 8001 et conserve son volume/configuration séparés. La collection Postman `postman/Project35_WP1-WP7.postman_collection.json` utilise les 12 features du modèle et des payloads PSI conformes.

Vérifications WP8 du 20 novembre 2026 et contrôle stack répété le 2026-11-21 : `docker compose config --quiet` réussi pour profils par défaut et synthétique; `scripts/check_stack.py` sur le réseau Compose réussi pour FastAPI, Django OpenAPI et MLflow (Django métier `degraded` est attendu sans champion métier); le profil synthétique répond avec `project35-synthetic-churn@champion` version `1`. La suite au 20 novembre avait **48 passed**, couverture **76%** en 49,35 s. Après l'ajout TreeExplainer et son test API, la suite complète courante a **50 passed**, **77%** en 63,66 s et 5 avertissements de dépendances; voir `docs/wp8-delivery.md`. Health, contrat FastAPI, Swagger, model-info, prediction, SHAP local/global et PSI ont été exercés sur les services actifs. Le résumé non sensible est dans `reports/synthetic/wp8-smoke.json`; `reports/` est ignoré par Git.

WP1–WP8 sont ainsi exercés techniquement sur le profil synthétique. La livraison métier reste incomplète tant que le dataset officiel, sa provenance/licence, la cible, les seuils métier et les décisions d'acceptation ne sont pas confirmés. Les captures finales, présentation convenue avec l'enseignant et archive propre de remise restent à produire. Aucun score ou explication synthétique ne valide le modèle métier.
