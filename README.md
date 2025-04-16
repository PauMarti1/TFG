# Resultats
## Resultats Validació

## Ultims Canvis: StratifiedGroupKFold, crossEntropyLoss amb pesos, tot entrenat a 25 epoques, guardat a un arxiu npz totes les caracteristiques demanades.

| Arquitectura | AUC | Recall Benigne | Recall Maligne | Precision Benigne | Precision Maligne | F1-Score Benigne | F1-Score Maligne |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ViT 25 epochs | 0.7780 ± 0.0426 | 0.5992 ± 0.1162 | 0.7913 ± 0.0577 | 0.4549 ± 0.0587 | 0.8824 ± 0.0364 | 0.5176 ± 0.0525 | 0.8325 ± 0.0338 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ViT Preentrenat 25 epochs | 0.6018 ± 0.0762 | 0.6531 ± 0.3104 | 0.5504 ± 0.2310 | 0.2544 ± 0.1135 | 0.8765 ± 0.0695 | 0.3600 ± 0.1595 | 0.6426 ± 0.1584 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ViT + GAT (Només matriu) | 0.6631 ± 0.0759 | 0.0186 ± 0.0228 | 0.9873 ± 0.0139 | 0.3000 ± 0.4000 | 0.7867 ± 0.0081 | 0.0348 ± 0.0427 | 0.8757 ± 0.0102 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ViT + GAT (Matriu + Features) 25 epochs| 0.6934 ± 0.1339 | 0.2720 ± 0.0857 | 0.8979 ± 0.0447 | 0.4459 ± 0.1835 | 0.8162 ± 0.0309 | 0.3324 ± 0.1127 | 0.8544 ± 0.0283 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ViT + CNN (Features + Arquitectura) 25 epochs | 0.6755 ± 0.0406 | 0.5537 ± 0.1315 | 0.7974 ± 0.1033 | 0.4529 ± 0.0737 | 0.8680 ± 0.0346 | 0.4846 ± 0.0530 | 0.8265 ± 0.0616 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ViT + CNN (Features + Linear) 25 epochs | 0.9077 ± 0.0870 | 0.8207 ± 0.1815 | 0.8487 ± 0.0627 | 0.6022 ± 0.1277 | 0.9456 ± 0.0530 | 0.6913 ± 0.1459 | 0.8936 ± 0.0527 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ViT + GCN (No Binarització) 25 epochs | 0.7720 ± 0.0963 | 0.3347 ± 0.1133 | 0.9082 ± 0.0577 | 0.5536 ± 0.2662 | 0.8304 ± 0.0370 | 0.4076 ± 0.1479 | 0.8669 ± 0.0405 |

## Resultats Holdout
| Arquitectura | AUC | Recall Benigne | Recall Maligne | Precision Benigne | Precision Maligne | F1-Score Benigne | F1-Score Maligne |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ViT | 0.2033 | 0.0400 | 0.8333 | 0.3333 | 0.2941 | 0.0714 | 0.4348 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ViT Preentrenat | 0.2300 | 0.4400 | 0.0833 | 0.5000 | 0.0667 | 0.4681 | 0.0741 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ViT + GAT (Només matriu) | 0.1567 | 0.0000 | 1.0000 | 0.0000 | 0.3243 | 0.0000 | 0.4898 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ViT + GAT (Matriu + Features) | 0.1650 | 0.0000 | 1.0000 | 0.0000 | 0.3243 | 0.0000 | 0.4898 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ViT + CNN (Features + Arquitectura) | 0.3200 | 0.1200 | 0.9167 | 0.7500 | 0.3333 | 0.2069 | 0.4889 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ViT + CNN (Features + Linear) | 0.3377 | 0.1905 | 0.9091 | 0.8000 | 0.3704 | 0.3077 | 0.5263 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ViT + GCN (No Binarització) | 0.4050 | 0.0400 | 0.8333 | 0.3333 | 0.2941 | 0.0714 | 0.4348 |

# Gràfiques del Loss:

## ViT:
![No carrega](Images/ViT25.png)

## ViT Preentrenat:
![No carrega](Images/ViTPreentrenat25.png)

## Vit + GNN (Matriu + Features):
![No carrega](Images/ViT+GNN25.png)

## Vit + CNN (Features + Arquitectura):
![No carrega](Images/ViT+CNN25.png)

## Vit + GCN (NoBin):
![No carrega](Images/ViT+GCN.png)
