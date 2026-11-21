# Plan de présentation — Project 35 (20 diapositives indicatives)

> Version de travail pour la démonstration synthétique approuvée le 2026-11-21. Identifier chaque score/visualisation comme synthétique; ne pas présenter ce plan comme la remise finale avant confirmation du format par le cours. Les emplacements de captures doivent être remplis avec de vraies captures d'exécution, jamais simulées.

Le nombre de 20 est un plan de travail, à confirmer avec l'enseignant. Les éléments ci-dessous sont rédigés à partir des preuves présentes. Les captures visuelles sont à prendre dans les services réels au moment de l'exposé; aucune capture fictive n'est revendiquée.

| # | Diapositive | Preuve/visuel à préparer |
|---:|---|---|
| 1 | Titre, équipe, cours | Project 35 — démonstration technique MLOps à données synthétiques; date 21-11-2026. Ajouter les noms de l'équipe à l'oral ou après confirmation du format. |
| 2 | Problème et valeur métier | Cas illustratif : classification du churn B2B simulé. La sortie montre le fonctionnement d'une chaîne ML; elle ne soutient aucune décision client réelle. |
| 3 | Exigences du sujet | Parcours livré WP1–WP8 : contrat/qualité, entraînement/MLflow, serving Django/Swagger, SHAP, PSI, gate, Docker et preuves. Distinguer les fonctions du sujet et les améliorations de gouvernance du cahier des charges. |
| 4 | Architecture MLOps | CSV versionné par hash → preflight/splits → pipeline sklearn → Tracking/Registry MLflow → API Django → SHAP/PSI → challenger et gate; FastAPI expose le contrat de données. Montrer les services Compose et leurs volumes séparés. |
| 5 | Provenance et contrat des données | Générateur `generator-v1-seed-3501`; 4 000 lignes d'entraînement; cible `churn`, positive 1; SHA-256 train `6167d68f…49ebbec`. Permission utilisateur limitée au synthétique, non au cours. |
| 6 | Qualité, EDA et risques de fuite | Preflight `ready_for_review`; zéro valeur nulle et doublon exact. Label dérivé d'une règle artificielle; risque principal : généralisation impossible au réel. Voir `reports/synthetic/phase1-data-quality.json`. |
| 7 | Splits sans fuite | Seed 35, split stratifié 2 400 train / 800 validation / 800 test. Holdout gate indépendant généré : 1 000 lignes, SHA-256 `7675b5d3…598abda0`; pas d'entités réelles ni temporelles. |
| 8 | Pipeline sklearn | Prétraitement numérique : médiane puis standardisation; catégories : imputation mode puis OneHotEncoder `handle_unknown=ignore`; quatre estimateurs testés. |
| 9 | Évaluation | Tableau des quatre candidats et métriques validation dans `docs/synthetic-demo.md`. LR : F1 0.2673, AUC 0.7654; aucun autre candidat n'améliore les deux critères à la fois. Ne pas afficher les scores comme performance métier. |
| 10 | MLflow Tracking | Montrer les runs `8713c1…`, `fdf58a…`, `6b6c96…`, `855949…`: hash dataset, seed/splits, paramètres, métriques et signatures/artefacts. MLflow est persistant via le volume Compose. |
| 11 | Model Registry | `project35-synthetic-wp1-5-20261121@champion` v1 pour le run LR dédié; `project35-synthetic-churn@champion` v1 conservé et `@challenger` v3 après FAIL. Ne pas confondre ces noms avec le modèle métier. |
| 12 | API et Swagger | FastAPI health/validation/docs; Django `/api/predict/`, `/api/model-info/`, `/api/schema/`, `/api/docs/`, `/api/explain/` et `/api/monitoring/psi/`. Parcourir la collection Postman WP1–WP7 avec payload synthétique. |
| 13 | SHAP local | Montrer une vraie réponse waterfall du smoke WP5 : 12 contributions, probabilité classe 1 et résidu additif `-5.6e-17`. Attribution prédictive, pas causalité. |
| 14 | SHAP global | Bar/beeswarm exécutés sur 3 observations synthétiques; fond de 100 lignes, SHA-256 `3e8d1317…ea8e32c4`. Les valeurs colorées des catégories sont des codes locaux, non des rangs. |
| 15 | PSI | Scénario stable : usage 0.0061, région 0.0053. Scénario artificiellement dérivé : usage 18.6433, région 41.7346. Fenêtres de 1 000 lignes; seuils/seuils d'alerte illustratifs seulement. |
| 16 | Retraining challenger | Rapport PSI drift `224da7cb…`; workflow relance Random Forest, trace le run et enregistre v2 puis v3 challenger. Un signal de dérive ne promeut pas directement le modèle. |
| 17 | Gate PASS/FAIL | Expliquer le FAIL contrôlé : F1 augmente mais AUC baisse; alias champion reste v1. Rapport v3 `1ae29d2e…`. La branche PASS est couverte par tests, mais aucune promotion PASS n'est revendiquée sur ce dataset. |
| 18 | Docker Compose et CI | Services MLflow 5000, FastAPI 8000, Django métier 8001, Django synthétique 8002; données montées en lecture seule et DB dédiée. `docker compose config` valide; Pytest courant : 50 passed, couverture 77%, 63.66 s, 5 warnings. |
| 19 | Limites, gouvernance et travaux restants | Synthetic scope uniquement; TreeExplainer est pris en charge pour les arbres, Permutation SHAP pour LR; format final à valider avec le cours; seuils PSI non métier; pas de déploiement production, étude de biais ou validation réelle. |
| 20 | Démonstration et conclusion | Parcours relançable : compose → Swagger/Postman → recherche run MLflow → SHAP/PSI → gate. Références : `docs/synthetic-demo.md`, `docs/wp8-delivery.md`, manifeste et rapports JSON synthétiques. |

## Matrice d'outils à prévoir si demandée

| Approche | Avantages à comparer | Limites/coûts à comparer |
|---|---|---|
| Fichier `.pkl` local | Simple pour un prototype mono-machine | Versionnage, sécurité, audit et partage à gérer manuellement |
| MLflow Tracking + Registry | Runs/artefacts/versioning/alias traçables et intégration au workflow | Serveur, base, artifact store, sauvegardes et contrôle d'accès à opérer |
| Service cloud propriétaire | Services managés, scalabilité et intégration à l'écosystème cloud | Coût, dépendance fournisseur, configuration/permissions et exigences de transfert des données |

Ne pas comparer des coûts financiers sans estimation datée, ni conclure à la conformité/sécurité d'un fournisseur sans analyse dédiée.
