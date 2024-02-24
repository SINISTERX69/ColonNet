"""
This script is used to predict images.
"""
from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt
import numpy as np

def display(img, output, threshold=0.5):
    """
    Displays the prediction of the model
    
    Parameters:
        img - Input image
        output - Model Predictions
        Threshold - Minimum value for positive classification
    """
    # Unpack Predictions
    label, bbox, annot = output
    label = label[0][0]
    bbox = bbox.squeeze()

    annot = annot.squeeze()
    annot = (annot > threshold).astype(int)
    
    # Get true labels using threshold
    true_label = 1 if (label > threshold) else 0

    # Calculate confidence scores for all predictions
    confidence_score = label if (true_label == 1) else 1 - label

    # Print Class prediction and Confidence score
    print("Bleeding") if (true_label == 1) else print("No Bleeding")
    print("Confidence Score", confidence_score)

    # Extract relevant coordinates
    min_point = (bbox[:2] * 224).astype(int)*true_label
    max_point = (bbox[2:] * 224).astype(int)*true_label

    # Convert color format to RGB
    img_rgb = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)

    # Draw bounding box on input image
    color = (255, 0, 0)

    # Initialize subplots
    fig, ax = plt.subplots(1, 3)

    # Display Input image
    ax[0].imshow(img_rgb)
    ax[0].set_title("Input Image")
    
    cv2.rectangle(img_rgb, tuple(min_point), tuple(max_point), color, thickness=2)
    # Display image with bounding box
    ax[1].imshow(img_rgb)
    ax[1].set_title("Bounding Box")

    # Display Segmentation Mask
    ax[2].imshow(annot*true_label, cmap='gray')
    ax[2].set_title("Segmentation Mask")

    for axi in ax:
        axi.axis('off')

    plt.tight_layout()
    plt.show()



model=load_model('ColonNet.h5')
impath=input("Enter the image path you want to predict: ")
img=cv2.imread(impath)
if img.any()==None:
    print("No file Found ")
else:
    img=img/255
    img=img.reshape(-1,224,224,3)
    yhat=model.predict(img)

    display(img[0],yhat)
