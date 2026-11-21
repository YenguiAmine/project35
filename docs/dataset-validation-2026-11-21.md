# Validation du dataset — 2026-11-21

## Portée

La validation exécutée porte sur le profil **généré synthétiquement** décrit dans `data/synthetic/manifest.json`. Ce profil est prêt pour les démonstrations logicielles, pas approuvé comme dataset métier ou dataset du cours. Aucun CSV métier n'est présent dans `data/raw/`, `data/processed/` ou `data/holdout/`.

## Contrôles exécutés

- Préflight avec `--require-approved` sur `configs/data_contract.synthetic.yaml` et `configs/data_provenance.synthetic.yaml` : résultat `ready_for_review`, hash vérifié.
- Population train : 4 000 lignes, 13 colonnes (12 features et `churn`), 0 valeur manquante, 0 doublon exact; classes `0: 3 165`, `1: 835`.
- Holdout gate : 1 000 lignes, mêmes features et cible; classes `0: 788`, `1: 212`; intersection de lignes exactes avec le train : 0.
- Splits internes train/validation/test : 2 400 / 800 / 800 lignes; deux classes présentes dans chacun.
- Hashes SHA-256 des six CSV d'entraînement, holdout, références SHAP/PSI et fenêtres PSI comparés au manifeste : **6/6 correspondent**.

Hash train : `6167d68f99c93fb74c89f0cf3da6866d94d2bb1fc29723178e8f4249c49ebbec`.

Hash holdout gate : `7675b5d3e4e02c9c298af1bf630843118ea4cbbecb4e90a8278bd651598abda0`.

## Décision

**Validation technique synthétique réussie. Validation métier non effectuée.** Le `approval_status: approved` dans la provenance synthétique est strictement limité au développement local; il ne signifie pas que l'enseignant ou un data owner a accepté ce jeu. Les labels sont simulés par le générateur et toutes les métriques obtenues avec ces CSV restent illustratives.

## Prochaine action requise

Obtenir le CSV et la définition de cible autorisés par le cours/data owner, ainsi que la provenance/licence et la politique de split. À réception, placer le fichier original dans `data/raw/`, créer la provenance et le contrat métier dédiés, lancer le préflight sans approbation, faire revoir les résultats, puis seulement passer à `--require-approved` et à l'entraînement.
