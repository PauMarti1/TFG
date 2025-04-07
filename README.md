# Resultats

## Resultats Training:
---
## Resultats Validació

| Arquitectura | AUC | Recall Benigne | Recall Maligne | Precision Benigne | Precision Maligne | F1-Score Benigne | F1-Score Maligne |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ViT | 0.9343 ± 0.0523 | 0.8580 ± 0.0730 | 0.8716 ± 0.1031 | 0.6818 ± 0.1498 | 0.9572 ± 0.0222 | 0.7509 ± 0.1153 | 0.9089 ± 0.0685 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ViT Preentrenat | 0.7117 ± 0.0533 | 0.7168 ± 0.3076 | 0.5788 ± 0.2024 | 0.2994 ± 0.1224 | 0.9045 ± 0.0653 | 0.4036 ± 0.1533 | 0.6803 ± 0.1223 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ViT + GNN (Només matriu) | 0.6631 ± 0.0759 | 0.0186 ± 0.0228 | 0.9873 ± 0.0139 | 0.3000 ± 0.4000 | 0.7867 ± 0.0081 | 0.0348 ± 0.0427 | 0.8757 ± 0.0102 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ViT + GNN (Matriu + Features)| 0.9448 ± 0.0986 | 0.8117 ± 0.2773 | 0.9800 ± 0.0341 | 0.8812 ± 0.2109 | 0.9539 ± 0.0664 | 0.8406 ± 0.2506 | 0.9664 ± 0.0507 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ViT + CNN (Features + Arquitectura)| 0.9455 ± 0.0748 | 0.8890 ± 0.1324 | 0.8952 ± 0.1032 | 0.7375 ± 0.1905 | 0.9657 ± 0.0399 | 0.7991 ± 0.1627 | 0.9265 ± 0.0739 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ViT + CNN (Features + Linear)| 0.8701 ± 0.0904 | 0.7817 ± 0.1375 | 0.8063 ± 0.0754 | 0.5354 ± 0.1067 | 0.9303 ± 0.0459 | 0.6322 ± 0.1133 | 0.8625 ± 0.0587 |

## Resultats Holdout
| Arquitectura | AUC | Recall Benigne | Recall Maligne | Precision Benigne | Precision Maligne | F1-Score Benigne | F1-Score Maligne |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ViT | 0.2800 | 0.0400 | 0.8333 | 0.3333 | 0.2941 | 0.0714 | 0.4348 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ViT Preentrenat | 0.2600 | 0.6400 | 0.0000 | 0.5714 | 0.0000 | 0.6038 | 0.0000 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ViT + GNN (Només matriu) | 0.1567 | 0.0000 | 1.0000 | 0.0000 | 0.3243 | 0.0000 | 0.4898 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ViT + GNN (Matriu + Features)| 0.3817 | 0.1200 | 0.9167 | 0.7500 | 0.3333 | 0.2069 | 0.4889 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ViT + CNN (Features + Arquitectura)| 0.3538 | 0.2400 | 0.9231 | 0.8571 | 0.3871 | 0.3750 | 0.5455 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ViT + CNN (Features + Linear)| 0.3377 | 0.1905 | 0.9091 | 0.8000 | 0.3704 | 0.3077 | 0.5263 |

# Gràfiques del Loss:

## ViT:
![No carrega](Images/ViTKFold.png)

## ViT Preentrenat:
![No carrega](Images/ViTPreentrenat.png)

## Vit + GNN (Matriu + Features):
![No carrega](Images/LossV+G.png)

## Vit + CNN (Features + Arquitectura):
![No carrega](Images/Vit+CNN(Arquitectura).png)

## Vit + CNN (Features + Lineal):
![No carrega](Images/Vit+CNN(Lineal).png)
