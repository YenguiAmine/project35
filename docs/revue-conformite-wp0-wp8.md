# Revue de conformité et feuille de route Project 35 (WP0–WP8)

Date de revue: 2026-11-20

## 1. Sources, autorité et portée de cette revue

Cette revue compare le code à deux documents fournis séparément. Le fichier `DS1_2026_2027_All_44_Projects_3BI_3BIS_Chaouki_Bayoudhi.pdf` est l’énoncé officiel; Project 35 se trouve aux pages imprimées 52–53 (pages PDF 53–54). Il décrit un exemple B2B Customer Churn & Credit Default Scoring Platform et les 4 étapes pratiques: (1) pipeline sklearn sans fuite + MLflow; (2) registre, alias @champion et service Django; (3) SHAP local + PSI; (4) retraining/workflow et gate champion/challenger.

Le PDF `cahier_charges_project35.pdf` est un cahier opérationnel de travail. Il rappelle à plusieurs endroits qu’il a été construit à partir des exigences discutées, que le texte officiel n’y est pas reproduit intégralement et que les valeurs (dataset, seuils, slides) sont à confirmer. Il aide à structurer le travail; une recommandation du cahier n’est pas automatiquement une exigence officielle. La présente revue sépare donc:
- **Exigence officielle vérifiée:** présente dans l’énoncé Project 35.
- **Contexte illustratif:** exemple de plateforme churn/default, sans schéma de données détaillé fourni.
- **Choix technique local:** décisions de ce dépôt, à garder configurables.
- **À confirmer:** détails absents des PDF, en particulier le dataset et les seuils exacts.

L’énoncé ne fixe pas de lien de dataset, de nom de cible, de classe positive ou de seuil numérique du gate. Le cahier demande de ne pas substituer un dataset librement choisi à celui de l’énoncé. En conséquence, aucun dataset n’a été téléchargé, aucun entraînement métier n’a été lancé et aucun score réel n’est annoncé.

## 2. Architecture et valeur ajoutée du MLOps

```text
Source de données approuvée
  → contrat / qualité / EDA / split sans fuite (WP1)
  → Pipeline sklearn unique: ColumnTransformer + imputation/encodage/scaling + estimateur
  → entraînement des candidats et métriques validation
  → MLflow Tracking: paramètres, hash, versions, métriques, artefacts/signature (WP2)
  → Model Registry: versions, source run, aliases challenger/champion (WP3)
  → Django REST: résolution alias, rechargement versionné, predict/proba (WP4)
  → SHAP local/global + waterfall/beeswarm (WP5)
  → PSI sur référence figée vs fenêtre courante et alerte (WP6)
  → retraining déclenché/planifié → candidat challenger
  → comparaison out-of-sample F1 + ROC-AUC → gate auditable (WP7)
  → si accepté: déplacement d’alias et hot reload; sinon champion conservé
  → CI, Compose, client API, captures, présentation et archive (WP8)
```

**Où commence le ML?** La préparation et l’ajustement d’une baseline commencent en WP1 dans `build_baseline_pipeline` et `run_phase1`. L’orchestration MLflow de l’entraînement est en WP2 dans `train_and_log`. Dans le workflow officiel, ces deux lots composent l’étape pratique 1. Le projet est MLOps dès WP0 pour l’environnement/configuration/versionnage, mais le parcours MLOps vertical (données → run → registre → serving → monitoring → retraining/gate) n’est complet qu’après WP7 et sa validation de bout en bout en WP8.

Valeur ajoutée: le pipeline évite que le prétraitement diffère entre entraînement et service; MLflow rattache modèle et données à un run; Registry contrôle la version servie; API teste le vrai modèle par contrat; SHAP rend les contributions inspectables; PSI détecte les changements de distribution; le gate empêche un candidat inférieur de remplacer le champion sans preuve.

## 3. État des lots et livrables existants

| Lot / étape | État vérifié | Fichiers importants et rôle | Valeur ajoutée et reste à faire |
|---|---|---|---|
| WP0 — Cadrage et environnement | Partiel/actif | `Pipfile`, `Pipfile.lock`, `pyproject.toml`, `.gitignore`, `Dockerfile`, `docker-compose.yml` | Pipenv et dépendances versionnées, SHAP ajouté. Confirmer jeu/cible/licence/seuils et rendre Compose autonome avec MLflow persistant avant démo. |
| WP1 — Données, qualité, split et baseline | Squelette implémenté, validation métier bloquée par absence de données | `src/project35/data/contract.py`: contrat/qualité; `src/project35/data/phase1.py`: split, ColumnTransformer/Pipeline et baseline; `scripts/phase1.py`; `configs/data_contract.example.yaml`; `tests/data/test_phase1.py` | Features numériques/catégorielles, imputations fit dans Pipeline, inconnues OneHot gérées. Manquent dataset réel, EDA métier, analyse de fuite validée, modèle(s) candidat(s), rapport métriques réel et preuve sur test final. |
| WP2 — MLflow Tracking | Implémenté; 4 candidats synthétiques comparés | `src/project35/training/trainer.py`, `scripts/train.py`, `configs/mlflow.example.yaml`, `tests/training/test_tracking.py`, `docs/synthetic-demo.md` | Trace hash dataset, paramètres, métriques, graphiques, rapport, signature et pipeline. Runs homogènes LR/RF/DecisionTree/ExtraTrees sur le split validation; le champion LR est conservé car aucun candidat n'améliore à la fois F1 et ROC-AUC. Pas de performance métier revendiquée. |
| WP3 — Registry et aliases | Implémenté, testé avec backend MLflow local | `src/project35/registry/service.py`, `scripts/register_model.py`, test E2E tracking | Versions attachées au run, tags et résolution alias champion/challenger; déplacement d’un champion existant exige maintenant un rapport gate PASS persisté et vérifiable. Aucun modèle métier n’est enregistré. |
| WP4 — Django REST / Swagger | Implémenté et testé; dépend d’un modèle/serveur MLflow réel | `django_app/predictions/views.py`, `serializers.py`, `services.py`, `config/urls.py`, test HTTP et E2E | Health, predict/proba, model-info, OpenAPI/Swagger; validation features, 503 registre, cache/rechargement versionné. Accès anonyme local seulement. Pas de collection Postman historique; ce client est livré ci-dessous. |
| WP5 — SHAP | Permutation + TreeExplainer implémentés et testés synthétiquement | `src/project35/explainability/shap_service.py`, `django_app/predictions/explainability.py`, `explain_serializers.py`, routes et tests | Local/global: contributions probabilité/base/résidu, images PNG, version modèle et SHA-256 référence. Auto TreeExplainer pour DecisionTree/RF/ExtraTrees avec regroupement one-hot vers les features sources; fallback Permutation pour LR. Attribution non causale; aucune performance métier ni UI de rendu standalone revendiquée. |
| WP6 — PSI / monitoring | Implémenté et testé synthétiquement | `src/project35/monitoring/psi.py`, `django_app/predictions/monitoring.py`, `scripts/monitor_psi.py`, route, tests, guide WP6 | Bins quantiles de référence figés, missing/OTHER/overflow gérés, rapport haché, alert stable/watch/drift et API/CLI. Un rapport drift valide peut déclencher WP7; référence, seuils métier et batch réel restent à fournir. |
| WP7 — Retraining, challenger et gate | Implémenté et testé synthétiquement | `src/project35/gating/service.py`, `scripts/run_challenger_workflow.py`, `configs/gate.example.yaml`, tests de gate/tracking, `docs/wp7-gate.md` | Comparaison F1/ROC-AUC sur un holdout commun, seuils stricts, rapport hashé et vérification registre avant promotion; déclenchement manuel ou PSI dérive. Aucune promotion réelle; dataset et marges métier à confirmer. |
| WP8 — Tests de livraison, CI et démonstration | Parcours synthétique vérifié | `.github/workflows/ci.yml`, `docker-compose.yml` (profil synthétique isolé), `scripts/check_stack.py`, collection Postman, `docs/wp8-delivery.md`, `presentation/outline-wp8.md` | Compose config valide; 49 tests passent (77% couverture), Django check propre, health/OpenAPI HTTP 200. Paquet technique ZIP vérifié sans données brutes, DB, modèle ou secrets; validation métier officielle reste hors du profil synthétique. |

## 4. Logique métier actuellement couverte

- **Contrat/data:** valide cible non nulle, colonnes requises, unicité de nom de colonnes, données non vides et plusieurs classes. Les valeurs/métadonnées officielles ne sont pas dans l’exemple.
- **Split:** stratégie stratifiée avec seed ou chronologique configurable; IDs exclus; dates converties en composantes. À valider sur le vrai problème (temps, entités répétées, distribution/classes).
- **Pipeline:** médiane + scaling pour numérique; imputation mode + OneHotEncoder ignore unknown pour catégoriel; estimateur LogisticRegression. Un test métier de fuite et la politique d’inclusion/exclusion doivent être approuvés.
- **Tracking:** apprentissage sur train; sélection/métriques sur validation; hash du CSV et artefacts MLflow. Le code ne doit pas utiliser le test pour sélectionner modèle/seuil.
- **Serving:** l’API résout `@champion`, détecte changement de version et recharge le pipeline; les entrées doivent correspondre aux `feature_names_in_`.
- **SHAP:** classe expliquée et échelle probabilité retournées; valeurs additive contrôlées par résidu; affichage explicitement non causal. Référence doit être colonnes features uniquement et jamais le test final. Le choix PermutationExplainer doit être réconcilié avec la mention TreeExplainer de l’énoncé et une interface de rendu visuel reste à faire si la démo attend une UI.
- **Gouvernance:** le service vérifie désormais un rapport PASS persisté dans le run challenger, avec versions/run correspondants, hash du jeu d’évaluation, features/classes concordantes et deux deltas au-dessus des marges. Les tests prouvent cette règle sur données synthétiques; ils ne constituent pas une promotion métier.

## 5. API et client de test

Importer `postman/Project35_WP1-WP7.postman_collection.json` dans Postman. Variables modifiables: FastAPI 8000, Django 8001, MLflow 5000, observation exemple et classe. La collection couvre:
1. FastAPI health, contrat, validate (données synthétiques explicitement marquées), OpenAPI — WP1.
2. MLflow UI joignable — WP2/WP3; le run et alias restent testés par pytest/CLI.
3. Django health, model-info, prédiction, OpenAPI — WP4.
4. Explication locale waterfall, globale bar/beeswarm — WP5.
5. Monitoring PSI d’un batch synthétique — WP6.

Pour model-info/predict/SHAP, il faut un MLflow actif, un alias champion, remplacer `sample_instance` par la vraie ligne de features, et configurer `PROJECT35_SHAP_BACKGROUND`. Une collection Postman ne remplace pas les tests pytest de training/registry: les appels d’entraînement et l’enregistrement sont des commandes de gestion, pas des endpoints HTTP métier.

Commandes documentées:
- Tout: `pipenv run pytest --cov=project35 --cov-report=term-missing`
- API ciblées: `pipenv run pytest tests/api`
- Django: `pipenv run python django_app/manage.py check`
- Swagger FastAPI: `http://127.0.0.1:8000/docs`
- Swagger Django: `http://127.0.0.1:8001/api/docs/`
- MLflow UI: `http://127.0.0.1:5000`

### Rôles d’équipe suggérés

| Rôle | Lots/responsabilités | Fichiers et preuves |
|---|---|---|
| Data owner / Data engineer | source, licence, cible, contrat, qualité, dérive/référence | `configs/data_contract.yaml`, `src/project35/data/`, WP6, fiche provenance |
| Data scientist / ML engineer | EDA, splits, baseline/candidats, métriques, SHAP interprétation | `src/project35/data/phase1.py`, `training/trainer.py`, `explainability/`, rapports |
| MLOps engineer | environnement, Tracking, Registry, aliases, gate, packaging/Compose | `Pipfile*`, `training/`, `registry/`, scripts, `docker-compose.yml` |
| Backend/API engineer | Django REST, validation, rechargement modèle, OpenAPI, Postman | `django_app/predictions/`, `django_app/config/urls.py`, collection Postman |
| QA / project lead | traçabilité exigences→tests→preuves, CI, démo, slides et archive | `tests/`, `.github/workflows/ci.yml`, captures et dossier de livraison |

## 6. Feuille de route restante — une étape après l’autre

### Étape 0 — Verrouiller les décisions et la provenance
1. Lire la consigne enseignant complète et confirmer si le sujet autorise un dataset public choisi par l’équipe.
2. Choisir un seul cas pour le modèle démo (churn ou défaut) et définir cible, sémantique, classe positive, métrique F1 average, coûts d’erreur, split, seuil F1/AUC et fenêtres PSI.
3. Vérifier licence, provenance, colonnes sensibles, fuite temporelle et droit de redistribution.
4. Geler `configs/data_contract.yaml`, puis enregistrer version/hash/licence/source.
**Livrable:** décision dataset signée, data contract rempli, citation/licence, critères gate/PSI versionnés.

### Étape 1 — Finir le WP1 et le début du flux ML
1. Placer le CSV autorisé dans `data/raw/` sans committer les données si licence/protection l’interdit.
2. Produire rapport qualité: types, manquants, doublons, cardinalités, classe, anomalies, exclusions, fuite.
3. Revoir et approuver chaque feature et la cible métier.
4. Construire split train/validation/test sans fuite temporelle ou par client; vérifier présence de classes.
5. Entraîner baseline et au moins les candidats demandés (mêmes splits et transformations Pipeline).
6. Choisir candidat et seuil uniquement sur validation; évaluer test final une seule fois.
**Livrable:** rapport EDA/qualité, Pipeline sérialisé, comparaison candidats et métriques validation/test avec protocole.

### Étape 2 — Compléter WP2/WP3 avec résultats réels
1. Lancer MLflow durable (tracking DB + artifact store) et créer experiment.
2. Tracer chaque candidat avec commit, dataset SHA, seed, split, hyperparamètres, métriques, signatures, confusion matrix et courbes.
3. Comparer runs, enregistrer version retenue; documenter run/version.
4. Initialiser alias champion et challenger sans ambiguïté; capturer UI.
**Livrable:** au moins deux runs réels comparables, Registered Model, aliases et preuve que version correspond au run/dataset.

### Étape 3 — Terminer WP4/WP5 de bout en bout
1. Charger alias via le service; vérifier modèle actif, schema et réponse 503 sans registre.
2. Exécuter client Postman sur modèle réel avec exemples anonymisés.
3. Configurer CSV de référence SHAP depuis train approuvé; vérifier schéma/types, classe et additivité.
4. Sauvegarder vrai waterfall local et bar+beeswarm global; relier artifacts à la version modèle et au hash référence.
5. TreeExplainer pour pipelines arbres et mapping post-ColumnTransformer vers features lisibles implémentés; PermutationExplainer reste utilisé pour le champion Logistic Regression selon compatibilité modèle.
6. Vérifier la définition métier de « waterfall pour chaque scored customer »: l’API locale existe séparément du scoring; l’intégrer à la réponse ou prouver le chaînage dans le client si une explication accompagne chaque score.
7. Si une UI locale est attendue, afficher les PNG décodés par observation au lieu de laisser les images en chaînes Base64 dans Swagger/Postman.
**Livrable:** API répondant sur vrai alias, OpenAPI/collection passante, choix explainer justifié, captures d’explications réelles.

### Étape 4 — Valider WP6 sur la référence approuvée
1. Faire approuver la fenêtre de référence et son hash, ainsi que les features surveillées.
2. Faire confirmer les seuils par le sujet/encadrant; les defaults 0.10/0.25 restent indicatifs.
3. Créer un batch courant anonymisé et valider les bins figés, OTHER, missing et overflow.
4. Capturer les rapports stable/watch/drift; connecter le signal au retraining WP7 seulement après validation.
**Livrable restant:** rapport PSI métier et preuve d’alerte; module/API/CLI et tests synthétiques sont livrés.

### Étape 5 — Valider WP7 challenger/gate/workflow implémenté

Le code, le rapport de décision et les tests synthétiques sont maintenant livrés; les actions ci-dessous décrivent le passage à la validation métier et la démonstration réelle. Voir [docs/wp7-gate.md](wp7-gate.md) pour les commandes et les règles du gate.
1. Déclencher par dérive (ou commande contrôlée) l’entraînement d’un candidat sur protocole fixé.
2. Tracer run puis l’affecter à @challenger sans toucher @champion.
3. Évaluer champion et challenger sur même dataset indépendant, même cible, classe positive, F1 average et calcul ROC-AUC.
4. Appliquer opérateurs/marges configurés et tests techniques; ne jamais recalculer seuil après avoir vu le score.
5. Générer rapport d’audit: versions/runs, hashes, scores, critères, raisons, date et décision.
6. En cas de PASS seulement, déplacer @champion par le service gouverné avec report signé/identifié; en FAIL garder alias.
7. Tester promotion pass, rejet inférieur, métrique manquante, modèle incompatible, gate report absent, alias inaccessible et hot reload Django.
8. Générer la narrative d’explication exigée uniquement à partir des valeurs numériques SHAP exactes; vérifier mécaniquement que les facteurs cités, signes et ordre correspondent aux valeurs et n’énoncent aucune causalité.
**Livrable:** script/commande workflow, rapport comparatif, décision auditée, tests pass/fail et capture du registre avant/après.

### Étape 6 — Compléter et démontrer WP8

Le squelette de livraison est livré dans [docs/wp8-delivery.md](wp8-delivery.md); Compose orchestre maintenant MLflow, FastAPI et Django avec volumes persistants, et la CI valide pytest, Django et la syntaxe Compose. La liste ci-dessous est le travail de validation/remise restant, pas du code non commencé.
1. Ajouter E2E: train → MLflow → Registry → predict → SHAP; référence/current → PSI → alerte; challenger → gate → promote/refuse.
2. Exécuter pytest complet en CI; ajouter gestion d’erreurs/performance et assertions schema.
3. Rendre Docker Compose reproductible avec MLflow, volumes de tracking/artifacts et configuration data; lancer réellement images et appels Postman.
4. Capturer résultats réels: runs, registry, scoring, SHAP local/global, PSI, gate.
5. Préparer présentation PowerPoint: définitions MLOps/training-serving skew/Tracking/Registry/PSI/math SHAP, installation, deux diagrammes demandés, matrice MLflow vs fichiers .pkl vs cloud propriétaire, résultats et limites. Vérifier nombre exact de slides auprès du guide enseignant; le cahier propose une cible 20 mais la marque à confirmer.
6. Fournir README reproductible, documentation API/architecture, rapports, citations/licences, PPTX+PDF et archive sans secrets/données interdites.
**Livrable:** paquet de démonstration reproductible, preuves, slides, rapport tests et archive finale.

## 7. Recherche de datasets — candidats, pas décision d’ingestion

L’énoncé décrit le churn et le crédit comme exemple de plateforme, mais ne fournit pas dans ses pages Project 35 un dataset imposé ni des variables métier. La recherche web ne constitue donc pas une autorisation de remplacer le jeu de données du cours.

| Cas possible | Source trouvée | Schéma/provenance visible | Usage et réserve |
|---|---|---|---|
| Churn télécom | [Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) | Chaque ligne est un abonné; services, contrat, tenure, charges, churn du dernier mois. | Bon benchmark de classification tabulaire et mixed features; ce n’est pas nécessairement B2B, vérifier les droits et la licence sur la page avant usage. |
| Churn télécom miroir | [Hugging Face telco-churn-7k](https://huggingface.co/datasets/temporaldrift777/telco-churn-7k) | 7 043 lignes, 21 champs, cible Churn; licence indiquée ECL-2.0. | Utile pour inspecter schéma et prototyper sous réserve de lire cette licence; miroir communautaire, vérifier sa provenance et éviter de prétendre qu’il s’agit d’un dataset B2B. |
| Défaut de carte | [Kaggle Credit Card Default Risk Analysis](https://www.kaggle.com/competitions/credit-card-default-risk-analysis/data) | Default of credit card clients de Taiwan, variables démographiques, crédit, paiements et factures; binaire Yes/No. | En compétition, inspecter règles d’accès, licence et disponibilité des labels train; données crédit sensibles et contexte géographique/historique non équivalent au B2B visé. |
| Défaut de carte, source de référence | [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default%2Bof%2Bcredit%2Bcard%2Bclients) | 30 000 observations, 23 features, classification; UCI indique CC BY 4.0 et citation DOI 10.24432/C55S3H. | Alternative proprement documentée si l’enseignant accepte ce cas; elle n’est pas une preuve que ce dataset est imposé. |

Recommandation de travail: attendre la confirmation du sujet/dataset. Si un jeu public est expressément autorisé, choisir un seul cas et figer cible/version/licence; ne pas mélanger churn et défaut dans un modèle unique avant d’avoir une définition de cible et une justification métier. Le HF mirror affiche ECL-2.0; cela mérite un contrôle de licence distinct avant redistribution.

## 8. Résultats de tests et critères de fin actuels

Référence de vérification antérieure à WP7/WP8: **34 tests pytest passés**, couverture 76%; `manage.py check` sans problème; commande CLI PSI --help validée; collection Postman WP1–WP6 JSON analysée; `docker compose config --quiet` valide. Ces résultats historiques ne valident pas les changements WP7/WP8. Une syntaxe Compose valide ne prouve pas que les images démarrent.

Vérification WP8 du 20 novembre 2026: le smoke-check réseau valide FastAPI, Django OpenAPI et MLflow; le profil `synthetic-demo-api` tourne sur 8002 avec `project35-synthetic-churn`, sans réutiliser le volume métier de 8001. Django métier répond `degraded`/`model_loaded=false`, attendu sans champion métier. `manage.py check` est propre; la collection Postman est valide; appels HTTP FastAPI/serving/SHAP/PSI réussis. La suite complète donne **48 passed**, couverture **76%**, 5 avertissements de dépendances. Les configurations Compose par défaut et profil synthétique passent. Ces tests ne sont pas une évaluation métier et ne remplacent pas les preuves sur dataset approuvé. Détails et commandes: [docs/wp8-delivery.md](wp8-delivery.md).

Le projet a une **stack prototype WP1–WP8 construite et démarrée**, avec vérification synthétique complète jusqu'au gate et client de démonstration isolé. La réalisation métier reste incomplète: dataset/provenance/contrat, seuils approuvés, démonstration sur données autorisées, captures, présentation conforme aux consignes du cours et paquet final doivent encore être confirmés et documentés.

