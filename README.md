# ColonNet
ColonNet is a Deep Learning Model that detects and highlights the instances of bleeding in the Gastrointestinal Tract. 

This project is a submission for The Auto-WCEBleedGen Challenge Ver2 by **Team ColonNet**.

## Network Architecture

![ColonNet_Architecture](Architecture_Images/ColonNet_Architecture.png)


![Double_Conv](Architecture_Images/Double_Conv.png)

The Neural Network consists of three branches for classification , segmentation and bounding box prediction.

### Classification & Bounding Box Prediction Branch

- DenseNet121 is used to extract features from the input image. This output is Maxpooled and flattened to pass into the classification branch.For the bounding box 
  branch, the output is Averagepooled and flattened.

- The classification branch consists of 5 fully connected layer connected by ReLU activation function. Dropout layers of 0.3 and 0.2 are implemented after first and
  second layer. Sigmoid Function is applied on the final layer which has only 1 node.

- The bounding box prediction branch consists of 6 fully connected layers connected by ReLU and ELU activation function. Dropout layer of 0.3 is implemented after 
  4th layer. Sigmoid Function is applied on the final layer to crunch the values of the coordinates between 0 and 1.

### Segmentation Branch

For Semantic Segmentation we have employed the traditional UNet architecture with Batch Normalization and ConvTranspose layers. It comprises of an Encoder path and a Decoder path which generates segmenation masks.

## Loss Functions

- Classification Branch - **Binary Cross Entropy Loss**
- Bounding Box Prediction Branch - **Mean Squared Error Loss**
- Segmentation Branch - **focal tversky Loss**

## Training Pipeline

**AdamW** Optimizer is used for training classification and bounding box branches and **Adam** optimizer is used for training segmentation branch.

1. We trained the Bounding Box branch for 10 epochs by feeding only bleeding images to the model. This resulted in **0.1806** validation loss.
2. Next we froze the parameters for the bounding box branch and the DenseNet and trained the model again for 10 epochs this time with the entire training dataset including non-bleeding images. At the end the validation loss for classification was **0.001**.
3. For segmentation branch, again only bleeding images were passed to the model for 30 epochs. The best validation loss obtained was **0.28**.

## PREDICTIONS

### VALIDATION DATASET 

| Bounding Box Prediction | CAM PLOT | Segmentation Mask |
| --- | ---- | --- |
| ![BOX_100](VAL/IMG100-BOX.png) | ![CAM_100](VAL/IMG100-CAM.png) | ![MASK_100](VAL/IMG100-MASK.png) |
| ![BOX_1044](VAL/IMG1044-BOX.png) | ![CAM_1044](VAL/IMG1044-CAM.png) | ![MASK_1044](VAL/IMG1044-MASK.png) |
| ![BOX_139](VAL/IMG139-BOX.png) | ![CAM_139](VAL/IMG139-CAM.png) | ![MASK_139](VAL/IMG139-MASK.png) |
| ![BOX_440](VAL/IMG440-BOX.png) | ![CAM_440](VAL/IMG440-CAM.png) | ![MASK_440](VAL/IMG440-MASK.png) |
| ![BOX_475](VAL/IMG475-BOX.png) | ![CAM_475](VAL/IMG475-CAM.png) | ![MASK_475](VAL/IMG475-MASK.png) |

### TEST DATASET 1

| Bounding Box Prediction | CAM PLOT | Segmentation Mask |
| --- | ---- | --- |
| ![BOX_25](TD1/TD1-A0025-BOX.png) | ![CAM_25](TD1/TD1-A0025-CAM.png) | ![MASK_25](TD1/TD1-A0025-MASK.png) |
| ![BOX_26](TD1/TD1-A0026-BOX.png) | ![CAM_26](TD1/TD1-A0026-CAM.png) | ![MASK_26](TD1/TD1-A0026-MASK.png) |
| ![BOX_27](TD1/TD1-A0027-BOX.png) | ![CAM_27](TD1/TD1-A0027-CAM.png) | ![MASK_27](TD1/TD1-A0027-MASK.png) |
| ![BOX_28](TD1/TD1-A0028-BOX.png) | ![CAM_28](TD1/TD1-A0028-CAM.png) | ![MASK_28](TD1/TD1-A0028-MASK.png) |
| ![BOX_31](TD1/TD1-A0031-BOX.png) | ![CAM_31](TD1/TD1-A0031-CAM.png) | ![MASK_31](TD1/TD1-A0031-MASK.png) |

### TEST DATASET 2

| Bounding Box Prediction | CAM PLOT | Segmentation Mask |
| --- | ---- | --- |
| ![BOX_152](TD2/TD2-A0152-BOX.png) | ![CAM_152](TD2/TD2-A0152-CAM.png) | ![MASK_152](TD2/TD2-A0152-MASK.png) |
| ![BOX_177](TD2/TD2-A0177-BOX.png) | ![CAM_177](TD2/TD2-A0177-CAM.png) | ![MASK_177](TD2/TD2-A0177-MASK.png) |
| ![BOX_194](TD2/TD2-A0194-BOX.png) | ![CAM_194](TD2/TD2-A0194-CAM.png) | ![MASK_194](TD2/TD2-A0194-MASK.png) |
| ![BOX_349](TD2/TD2-A0349-BOX.png) | ![CAM_349](TD2/TD2-A0349-CAM.png) | ![MASK_349](TD2/TD2-A0349-MASK.png) |
| ![BOX_361](TD2/TD2-A0361-BOX.png) | ![CAM_361](TD2/TD2-A0361-CAM.png) | ![MASK_361](TD2/TD2-A0361-MASK.png) |




## HOW TO USE
First make sure that your folder structure looks like the tree shown below

![sample-tree](https://github.com/SINISTERX69/ColonNet/assets/123566211/6899826b-7d08-4e22-aa98-1d687df52771)

Then install the requirments given in the **requirements.txt**

To train the model yourself you can simply run the **training.py**

To simply make predictions on images,run **prediction.py** and give the image path when asked. (You can also just give the image name but the image should be in the **same folder** as predictions.py)

## CREDITS

We wish to thank all the members of MISAHUB for organizing this challenge and providing the relevant image dataset for the training and testing of this model.

For further information and model metrics kindly refer to the README.pdf and excel file.
