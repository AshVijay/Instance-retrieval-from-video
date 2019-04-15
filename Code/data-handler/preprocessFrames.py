#!/usr/bin/env python
# -*- coding: utf-8 -*-


import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision.models.inception import inception_v3
from pymongo import MongoClient
import json
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
#from sklearn.decomposition import PCA
#from sklearn.cluster import KMeans




def getFeatures():
    """
    Driver function for preprocessing(embedding extraction, PCA, KMeans)
    """
    dataSet = MyDataset() 

    #Define the model
    base_model = inception_v3(pretrained=True)
    #print(base_model)
    base_model = nn.Sequential(*list(base_model.children())[:-4])  #Extract features from final max pool layer
    #print(base_model)
    
    dataset_loader = torch.utils.data.DataLoader(dataSet, batch_size = 1 , shuffle = True, num_workers = 4)
    train_iter = iter(dataset_loader)
    while(1):
      try:	
        images, labels = train_iter.next()
        images= Variable(images)
        #print(images.shape)
      except :
      	print("Reached end of data set")
      	break

      #Transform the image
      n,w,h,c=images.shape
      images.resize_(n,c,h,w)
      print(images.shape)
      print(labels)

      output = base_model(images)
      print(output.shape)

    
def Rescale(image, width, height):
    """Rescale the proposals to a given fixed size
    """
    return cv.resize(image,(width,height))

def Crop(image,x,y,w,h) :
    """Crop the bounding box
    """
    return image[y:y+h,x:x+w]
    
    
def KMeans():
	"""
	Function to reduce spatial dimensions of features
	"""
	return True

def PCA():
	"""
	Function to reduce individual dimension of feature vectors extracted from NN
	"""
	return True


class MyDataset(Dataset):
    """
    Custom class for converting dataset to torch format
    """
    def __init__(self, transform=None):
    	
        client = MongoClient("mongodb://localhost:27017/")
        mydatabase = client['InstanceRetrieval']
        collection = mydatabase['Frame']
        db_iterable = collection.find()
        

        #To get total number of images including proposals
        N= 0

        for document in db_iterable:
          bounding_boxes = document["Bounding_Boxes"]
          for proposal in bounding_boxes:
          	N+= 1
        

        images = np.zeros((N,128,128,3),dtype = np.uint8)
        labels = [] #To store the labels for each image as the corresponding Frame number
        
        n=0

        db_iterable = collection.find()
        for document in db_iterable:
          bounding_boxes = document["Bounding_Boxes"]	
          for proposal in bounding_boxes: 
            print("Reading document and bounding_box",proposal)
            print(bounding_boxes[proposal])

            try:  #Some bounding boxes have inconsistent sizes....needs to be checked
               x,y,w,h=bounding_boxes[proposal][0],bounding_boxes[proposal][1],bounding_boxes[proposal][2],bounding_boxes[proposal][3]
            except Exception as e:
            	continue
            	print(str(e))

            img_path= os.path.join(document["Video"],document["Frame"])
            temp_image = cv.imread(img_path)
            print("Read Image")
            temp_image = Crop(temp_image,x,y,w,h)
            images[n] = Rescale(temp_image,128,128)
            
            #Display figures 
            #plt.imshow(images[n])
            #plt.show(block=False)
            #plt.pause(1)
            #plt.close('all')

            labels += [os.path.join(str(img_path.split("/")[-1:-3:-1][1]) , str(img_path.split("/")[-1:-3:-1][0]) )]
            n+=1

        self.images = torch.from_numpy(images).float()
        self.labels = labels

        
    def __getitem__(self, index):
        """
        Retrieve indexed indexed image

        """
        image = self.images[index]
        frame = self.labels[index]

        return image, frame
    
    def __len__(self):
        return len(self.labels)



if ( __name__ == "__main__"):
         getFeatures()