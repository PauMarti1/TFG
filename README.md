# Descripció fitxers:

A tots els entrenos que fan ús de Graph Neural Networks he fet servir features i edge attributes:

RocCurves: Imatges de totes les corbes ROC resultants d'entrenar els models.

Images: Graelles de loss de tots els entrenos que he anat fent

Execucions: Fitxers finals de les arquitectures basades en graf.

NousCodis: Penúltims codis que es van fer servir per provar arquitectures.

Tradicionals: Fitxers finals de les arquitectures no basades en grafs.

Attention_triplets: Triplets on es mostra la imatge original, on posa l’atenció el ViT i on posa l’atenció el GAT.

DenseNet.py (DenseNet 25 epochs): KFold i Holdout de una Densenet

EnsembleGAT.py (ViT + GAT (Ensemble) 25 epochs): Extreure les 12 matrius d’atenció del ViT, entrenar un GAT amb cada una d’aquestes matrius i fer MaxVoting en el Holdout.

EnsembleGCN.py (ViT + GCN (Ensemble) 25 epochs): Extreure les 12 matrius d’atenció del ViT, entrenar un GCN amb cada una d’aquestes matrius i fer MaxVoting en el Holdout.

GATAGG.py (ViT + GAT (Agregació) 25 epochs): Agafar les 12 matrius d’atenció del ViT, passar-les a una capa d’agregació i fer-les servir per entrenar el model GAT.

GCNAgg.py (ViT + GCN (Agregació) 25 epochs): Agafar les 12 matrius d’atenció del ViT, passar-les a una capa d’agregació i fer-les servir per entrenar el model GCN.

MatAdj+Features.py (ViT + GAT (Mitjana) 25 epochs): Agafar les 12 matrius d’atenció del ViT, fer-ne la mitjana i fer servir la matriu resultant de la mitjana per entrenar un model GAT.

ResNet.py (ResNet 25 epochs): KFold i Holdout de una Resnet.

ViT+CNN (Lineal).py (ViT + CNN (Linear) 25 epochs): Agafar el vector de característiques del ViT, fer-ne la mitjana i fer servir la matriu resultant de la mitjana per entrenar una capa Linear.

ViT+CNNArc.py (ViT + CNN (Arquitectura) 25 epochs): Agafar el vector de característiques del ViT, fer-ne la mitjana i fer servir la matriu resultant de la mitjana per entrenar la arquitectura CNN que em vas passar per mail.

ViT+GCN.py (ViT + GCN (Binarització) 25 epochs i ViT + GCN (No Binarització) 25 epochs): Agafar les 12 matrius d’atenció del ViT, fer-ne la mitjana i fer servir la matriu resultant de la mitjana per entrenar un model GCN.

ViTKFold.py (ViT 25 epochs): KFold i Holdout de un ViT tal cual el dona la llibreria.

ViTPreentrenat.py (ViT Preentrenat 25 epochs): KFold i Holdout de un ViT amb els pesos d’un ViT suposadament entrenat en histopatologia.

heatmaps.py: Procés de creació dels mapes de calor per visualitzar la atenció de cada model.

CNN.py (CNN 3 Conv2d 50 epochs): Entreno validació i holdout utilitzant una cnn amb 3 capes de convolució.

# Resultats:
| Arq/Mètrica       | AUC               | RecB              | RecM              | PrecB             | PrecM             | F1-B              | F1-M              |
|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|
| CNN               | 0.7011 ± 0.0692   | 0.7492 ± 0.2008   | 0.4100 ± 0.1918   | 0.6155 ± 0.0768   | 0.4778 ± 0.1594   | 0.6273 ± 0.0928   | 0.3809 ± 0.1222   |
| ResNet            | 0.9279 ± 0.0441   | 0.8222 ± 0.1058   | 0.8753 ± 0.0665   | 0.8990 ± 0.0424   | 0.8110 ± 0.1043   | 0.8528 ± 0.0514   | 0.8359 ± 0.0608   |
| DenseNet          | 0.4996 ± 0.0009   | 1.0000 ± 0.0000   | 0.0000 ± 0.0000   | 0.5423 ± 0.0691   | 0.0000 ± 0.0000   | 0.7006 ± 0.0602   | 0.0000 ± 0.0000   |
| ViT               | 0.9113 ± 0.0593   | 0.8242 ± 0.1026   | 0.8876 ± 0.0493   | 0.8970 ± 0.0425   | 0.7948 ± 0.0917   | 0.8438 ± 0.0582   | 0.8361 ± 0.0637   |
| ViT transpath     | 0.7594 ± 0.0742   | 0.7459 ± 0.2131   | 0.7730 ± 0.1734   | 0.8315 ± 0.0933   | 0.7622 ± 0.1250   | 0.7547 ± 0.1430   | 0.7453 ± 0.0869   |
| ViTNNLin          | 0.9537 ± 0.0771   | 0.8989 ± 0.1070   | 0.9346 ± 0.0538   | 0.9461 ± 0.0410   | 0.8875 ± 0.1212   | 0.9194 ± 0.0724   | 0.9072 ± 0.0848   |
| ViTNNArq          | 0.8286 ± 0.0460   | 0.7633 ± 0.0968   | 0.8939 ± 0.0742   | 0.9057 ± 0.0505   | 0.7679 ± 0.0702   | 0.8232 ± 0.0597   | 0.8233 ± 0.0553   |
| ViTGAT (M) P      | 0.9000 ± 0.0696   | 0.7385 ± 0.1056   | 0.9469 ± 0.0305   | 0.9473 ± 0.0290   | 0.7581 ± 0.0916   | 0.8253 ± 0.0668   | 0.8390 ± 0.0613   |
| ViTGAT (A) P      | 0.9411 ± 0.0415   | 0.7523 ± 0.1011   | 0.9530 ± 0.0235   | 0.9516 ± 0.0156   | 0.7738 ± 0.0767   | 0.8366 ± 0.0647   | 0.8515 ± 0.0488   |
| ViTGAT (E) P      | 0.9353 ± 0.0427   | 0.7774 ± 0.0885   | 0.9450 ± 0.0249   | 0.9468 ± 0.0167   | 0.7874 ± 0.0768   | 0.8513 ± 0.0573   | 0.8573 ± 0.0538   |
| ViTGAT (M)        | 0.9233 ± 0.0404   | 0.8031 ± 0.1257   | 0.8915 ± 0.0887   | 0.9162 ± 0.0545   | 0.8162 ± 0.1089   | 0.8476 ± 0.0622   | 0.8430 ± 0.0565   |
| ViTGAT (A)        | 0.9000 ± 0.0531   | 0.7779 ± 0.1079   | 0.9096 ± 0.0551   | 0.9190 ± 0.0439   | 0.7863 ± 0.0865   | 0.8365 ± 0.0582   | 0.8389 ± 0.0507   |
| ViTGAT (E)        | 0.9340 ± 0.0349   | 0.7632 ± 0.1103   | 0.9384 ± 0.0253   | 0.9400 ± 0.0188   | 0.7806 ± 0.0834   | 0.8381 ± 0.0698   | 0.8498 ± 0.0555   |
| ViTGCN (M) P      | 0.9115 ± 0.0265   | 0.7406 ± 0.0560   | 0.9091 ± 0.0567   | 0.9095 ± 0.0439   | 0.7463 ± 0.0639   | 0.8157 ± 0.0474   | 0.8191 ± 0.0586   |
| ViTGCN (A) P      | 0.9148 ± 0.0288   | 0.7399 ± 0.0722   | 0.9431 ± 0.0271   | 0.9448 ± 0.0249   | 0.7542 ± 0.0757   | 0.8275 ± 0.0421   | 0.8363 ± 0.0518   |
| ViTGCN (E) P      | 0.9367 ± 0.0387   | 0.7832 ± 0.0904   | 0.9358 ± 0.0250   | 0.9379 ± 0.0161   | 0.7911 ± 0.0727   | 0.8510 ± 0.0577   | 0.8558 ± 0.0496   |
| ViTGCN (M)        | 0.9076 ± 0.0542   | 0.7095 ± 0.1513   | 0.9435 ± 0.0606   | 0.9534 ± 0.0340   | 0.7538 ± 0.0996   | 0.8017 ± 0.0882   | 0.8313 ± 0.0529   |
| ViTGCN (A)        | 0.9238 ± 0.0441   | 0.7963 ± 0.1212   | 0.9054 ± 0.0488   | 0.9161 ± 0.0155   | 0.8110 ± 0.0926   | 0.8468 ± 0.0692   | 0.8509 ± 0.0478   |
| ViTGCN (E)        | 0.9279 ± 0.0359   | 0.7745 ± 0.0894   | 0.9393 ± 0.0234   | 0.9395 ± 0.0204   | 0.7845 ± 0.0760   | 0.8466 ± 0.0585   | 0.8532 ± 0.0524   |


