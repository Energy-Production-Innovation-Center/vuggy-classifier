# Classification of dissolution zones in BHI images
Author: Guilherme F. Chinelatto, João P. P. Souza, Mateus Basso

## Abstract
In recent years, advances in computer science and machine learning technologies, such as the convolutional neural network (CNN), have helped geoscientists solve many problems related to image classification.  CNN is a computational artificial network applied for computer vision that helps in object detection, image segmentation, classification, among others. Some of these applications have been aimed at classifying geological features in borehole images (BHI), achieving promising results. The dissolution features in carbonate rocks are very common, they may vary from millimetric pores, enlarged conduits, open and enlarged fractures and caves. The identification of these features are important in several areas, such as in the oil industry since dissolved zones in carbonate reservoirs impact the capacity to conduct fluids, in addition to helping the drilling campaign to predict zones with high losses of drilling mud. Based on this, this work proposes a novel methodology for the identification of dissolution features in carbonate rocks using BHI and CNN models. For this task, seven wells with acoustic BHI from carbonates of Barra Velha Formation located at Santos Basin, Brazil, were used. Initially the workflow involves defining the scale and the height size of the images, followed by class definition and dataset classification, CNN model training and finally applying the best model to a blind well to check the performance in an untrained dataset. The resolution of 1:10 was defined and four scenarios (S1-S4) of cut vertical height sample size (CHS) of 10 cm (S1), 40 cm (S2), 70 cm (S3) and 100 cm (S4) were chosen, and then the images were classified. Six classes were proposed according to the density of dissolution features where (0) is the non-dissolved carbonate rock and (5) are images with occurrence of large low-impedance features representing dissolved features such as conduits and caves. Intermediate (1) and (3) are the vuggy matrix and (2) and (4) fracture classes, varying their degree of density. Four CNN models were used for the prediction of dissolution features being them the ResNet, RegNet, ShuffleNet and MobileNet. For S1 to S3, the best performance was achieved by the ResNet model with accuracy values around 0.90 whereas for S4 MobileNet with values close to 0.70. A qualitative analysis was then made in the blind well to verify the performance of each model in an untrained/unseen dataset. As expected the results revealed a decreasing of the performance with the increasing of CHS but in general exhibited promising results when combining CHS of 10 and 70 cm (S1 and S3) identifying the main regions with occurrence of dissolution zones as also as the main trends of classes distributed along the well.
