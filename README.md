# Resultats
## RESULTATS ViT KFOLD (5FOLDS, 10 EPOCHS PER FOLD) Només entrenant la capa de classificació.

Resultats training:
AUC: 0.9343 ± 0.0523
Recall Benigne: 0.8580 ± 0.0730
Recall Maligne: 0.8716 ± 0.1031
Precision Benigne: 0.6818 ± 0.1498
Precision Maligne: 0.9572 ± 0.0222
F1-score Benigne: 0.7509 ± 0.1153
F1-score Maligne: 0.9089 ± 0.0685

Holdout results: 
AUC: 0.2800
Recall Benigne: 0.0400
Recall Maligne: 0.8333
Precision Benigne: 0.3333
Precision Maligne: 0.2941
F1-score Benigne: 0.0714
F1-score Maligne: 0.4348

## RESULTATS ViT Preentrenat en histopatologia KFOLD (5FOLDS, 10 EPOCHS PER FOLD) Només entrenant la capa de classificació.

Resultats Training:
AUC: 0.7117 ± 0.0533
Recall Benigne: 0.7168 ± 0.3076
Recall Maligne: 0.5788 ± 0.2024
Precision Benigne: 0.2994 ± 0.1224
Precision Maligne: 0.9045 ± 0.0653
F1-score Benigne: 0.4036 ± 0.1533
F1-score Maligne: 0.6803 ± 0.1223

Holdout results: 
AUC: 0.2600
Recall Benigne: 0.6400
Recall Maligne: 0.0000
Precision Benigne: 0.5714
Precision Maligne: 0.0000
F1-score Benigne: 0.6038
F1-score Maligne: 0.0000

## Resultats intent de ViT + GNN (GAT):

Resultats Training:
AUC: 0.6631 ± 0.0759
Recall Benigne: 0.0186 ± 0.0228
Recall Maligne: 0.9873 ± 0.0139
Precision Benigne: 0.3000 ± 0.4000
Precision Maligne: 0.7867 ± 0.0081
F1-score Benigne: 0.0348 ± 0.0427
F1-score Maligne: 0.8757 ± 0.0102

Resultats Holdout:
Recall Benigne: 0.0000
Recall Maligne: 1.0000
Precision Benigne: 0.0000
Precision Maligne: 0.3243
F1-score Benigne: 0.0000
F1-score Maligne: 0.4898
AUC: 0.1567

## Resultats ViT + GNN (Matriu d'adjacència + Features):

Resultats Training:
Recall Benigne: 0.8117 ± 0.2773
Recall Maligne: 0.9800 ± 0.0341
Precision Benigne: 0.8812 ± 0.2109
Precision Maligne: 0.9539 ± 0.0664
F1-score Benigne: 0.8406 ± 0.2506
F1-score Maligne: 0.9664 ± 0.0507
AUC: 0.9448 ± 0.0986

Resultats Holdout:
AUC: 0.3817
Recall Benigne: 0.1200
Recall Maligne: 0.9167
Precision Benigne: 0.7500
Precision Maligne: 0.3333
F1-score Benigne: 0.2069
F1-score Maligne: 0.4889

