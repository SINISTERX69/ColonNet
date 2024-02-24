import albumentations as alb
import cv2
import os
import numpy as np


def boxarea(box):
    """Returns the area of the box
    `box`:list/tuple of coordinates
    """
    return (box[2]-box[0])*(box[3]-box[1])

def biggestbbox(path):
    """
    returns the biggest box from a txt file with multiple box coordinates
    `path`:file path
    """
    with open(path) as f:
        boxes=list(map(int,f.readline().split()))
    bigbox=[0,0,0,0]
    bbarea=boxarea(bigbox)
    for i in range(0,len(boxes),4):
        box=boxes[i:i+4]
        if bbarea<boxarea(box):
            bigbox=box
            bbarea=boxarea(box)
    if bigbox==[0,0,0,0]:
      return [0],[0,0,1/224,1/224]
    return [1],[c/224 for c in bigbox]


def load_data(with_neg=True,as_array=True,aug=False,nums=5):
    
    """
    Loads data from predefined path with custom parameter for training classification and bounding boxes branches

    `with_neg`:include non-bleeding images
    `as_array`:returns np.array if set `True` else List
    `aug`: apply augmentation if set `True`
    `nums`: defines number of images to be made from 1 (needed when `aug` is `True`)
    """



    bpath="/WCEBleedGen (updated)/bleeding/Images/"
    bboxpath="/WCEBleedGen (updated)/bleeding/Bounding boxes/TXT/"

    images=[]
    boxes=[]
    ann=[]
    if aug:
        augmentor=alb.Compose([
            alb.BBoxSafeRandomCrop(),
            alb.HorizontalFlip(p=0.5),
            alb.VerticalFlip(p=0.5),
            alb.Rotate(),
            alb.Resize(height=224,width=224,p=1)],

            bbox_params=alb.BboxParams(format='albumentations',
                                      label_fields=['class_labels'])
    )
      
    for impath in os.listdir(bpath):
        img=cv2.imread(os.path.join(bpath,impath))
        img=img/255.0

        box_path=os.path.join(bboxpath,impath.split('.')[0]+".txt")
        num,box=biggestbbox(box_path)
        
        
        if aug:
            if boxarea(box)<0.125:
                t=nums+3
            elif boxarea(box)<0.25:
                t=nums+2
            else:
                t=nums
            for _ in range(t):
                augmented=augmentor(image=img,bboxes=[box],class_labels=[num])

                try:
                    boxes.append(augmented['bboxes'][0])
                    ann.append(augmented['class_labels'][0])
                    images.append(augmented['image'])
                except:
                    pass
                
        else:
            images.append(img)
            boxes.append(box)
            ann.append(num)
    if with_neg:
        nbpath="/WCEBleedGen (updated)/non-bleeding/images/"
            
        for impath in os.listdir(nbpath):
            img=cv2.imread(os.path.join(nbpath,impath))
            img=img/255.0
            if aug:
                for _ in range(nums):
                    augmented=augmentor(image=img,bboxes=[[0.,0.,1.,1.]],class_labels=[[0]])
                    
                    images.append(augmented['image'])
                    boxes.append([0,0,1/224,1/224])
                    ann.append([0])
                    
            else:
                images.append(img)
                boxes.append([0,0,1/224,1/224])
                ann.append([0])
    if not as_array:
        return images,boxes,ann
    return np.array(images),np.array(boxes),np.array(ann)


def load_test(image_path=None,labels=None):
    """ Load Test-Dataset 2 Images,classification and Labels from TXT."""

    image_path=image_path or "/Test Dataset for Auto-WCEBleedGen Challenge version 2/Test Dataset 2/Images/"
    labels=labels or "/Test Dataset for Auto-WCEBleedGen Challenge version 2/Test Dataset 2/Lables/TXT/"

    images=[]
    boxes=[]
    ann=[]
    for impath in os.listdir(image_path):
        img=cv2.imread(os.path.join(image_path,impath))
        img=img/255.0

        box_path=os.path.join(labels,impath.split('.')[0]+".txt")
        num,box=biggestbbox(box_path)

        images.append(img)
        boxes.append(box)
        ann.append(num)

    return np.array(images),np.array(boxes),np.array(ann)


def load_data_unet(aug=False,nums=5):

    """
    Loads only bleeding data from predefined path with custom parameter for training segmentation branch with images and masks

    `aug`: apply augmentation if set `True`
    `nums`: defines number of images to be made from 1 (needed when `aug` is `True`)
    """



    bpath="/WCEBleedGen (updated)/bleeding/Images/"
    bannpath="/WCEBleedGen (updated)/bleeding/Annotations/"

    images=[]
    ann=[]

    if aug:
        augmentor=alb.Compose([
            alb.HorizontalFlip(p=0.5),
            alb.VerticalFlip(p=0.5),
            alb.Rotate(),
            alb.Resize(height=224,width=224,p=1)
            ])
    for impath in os.listdir(bpath):
        img=cv2.imread(os.path.join(bpath,impath))
        img=img/255.0
        
        annpath='ann-'+impath.split("-")[-1]
        mask=cv2.imread(os.path.join(bannpath,annpath),0)
        mask=mask/255

        
        if aug:
            for _ in range(nums):
                augmented=augmentor(image=img,mask=mask)

                try:
                    images.append(augmented['image'])
                    ann.append(augmented['mask'])
                except:
                    count+=1
                    pass
                
        else:
            images.append(img)
            ann.append(mask)
            

    return np.array(images),np.array(ann)


def load_test_unet(pathim,pathann):
    """
    loads images and masks for testing from user specified paths
    
    `pathim`:path to image folder
    `pathann`:path to mask folder
    
    """


    files=os.listdir(pathim)
    files.sort()
    images=[]
    ann=[]
    for file in files:
        im=cv2.imread(os.path.join(pathim,file))
        im=im/255.0
        
        mask=cv2.imread(os.path.join(pathann,file),0)
        mask=mask/255
        
        images.append(im)
        ann.append(mask)
        
    return np.array(images),np.array(ann)
        