# Dataset synthétique Project 35

`b2b_churn_train.csv` et `b2b_churn_gate_evaluation.csv` sont générés localement par `scripts/generate_synthetic_dataset.py`. Ils ne contiennent aucune observation de client réel. Le holdout est produit avec un flux aléatoire indépendant du train; il sert uniquement à vérifier le fonctionnement du gate et des APIs.

Les étiquettes `churn` sont simulées à partir d'une formule probabiliste documentée dans le générateur. Les scores, explications SHAP, alertes PSI ou décisions du gate obtenus avec ces fichiers sont des démonstrations techniques, pas des résultats métier ou des résultats du cours. Le dataset officiel reste à fournir/valider.

`b2b_churn_shap_reference.csv` contient 100 lignes tirées de la population train avec seed 35 et uniquement les 12 features (sans cible). Ce sous-ensemble est prévu pour les démonstrations SHAP à faible coût; il ne constitue pas une référence métier approuvée.

Pour WP6, `b2b_churn_psi_reference.csv` contient les 4 000 lignes train sans cible; `b2b_churn_psi_stable_batch.csv` est le holdout indépendant sans cible; `b2b_churn_psi_drift_batch.csv` est une copie de démonstration dont `product_usage_ratio_30d` et `region` ont été décalés artificiellement. Les hashes et tailles sont listés dans `manifest.json`. Ces fenêtres servent à vérifier les chemins stable/alert, et ne représentent pas une dérive réelle.
