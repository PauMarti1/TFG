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
