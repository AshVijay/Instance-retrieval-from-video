#!/usr/bin/env python
# -*- coding: utf-8 -*-


import time
import torch
import torch.nn as nn
import json
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
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans




def getFeatures():
    """
    Driver function for preprocessing(embedding extraction, PCA, KMeans)
    """
    dataSet = Dataset_Primary() 

    #Configuration dump from json config file








    #Define the model
    base_model = inception_v3(pretrained=True)
    #print(base_model)
    base_model = nn.Sequential(*list(base_model.children())[:-4])  #Extract features from final max pool layer
    #print(base_model)
    
    dataset_loader = torch.utils.data.DataLoader(dataSet, batch_size = 1 , shuffle = False, num_workers = 4)
    train_iter = iter(dataset_loader)

    global_features = np.zeros(shape=(1,1000))
    global_labels = []

    while(1):
      try:	
        image, label = train_iter.next()   #Images and labels generated according to DataSet_Primary's class definition 
        image= Variable(image)
        #print(image.shape)
      except :
      	print("Reached end of data set")
      	break

      #Transform the image
      n,w,h,c=image.shape
      image.resize_(n,c,h,w)
   
      output = base_model(image)
      output = output.data.numpy()
      #print("Feature Vector:",output)
      #print("Label :", label)

      #Append features to create an M*1000 matrix for all M proposals
      global_features = np.concatenate([global_features,output], axis = 0)
      global_labels.append(label[0])

    data_dict = {}
    #Each element is of the form 'Video1':{  {'Frame1':np.array, 
    #                                                           'Frame2':np.array....
    #                                                           }
    #                                                        }
                          

    global_features = reduceDimensions(global_features)
    #print(global_features)
    #print(type(global_labels[0]))


    
    for index, (_,_) in enumerate(zip(global_features,global_labels)):
        video_name = global_labels[index].split("/")[0]
        frame_name = global_labels[index].split("/")[1]
        if(video_name not in data_dict.keys()):
        	data_dict[video_name] = {}
        
        if(frame_name not in data_dict[video_name].keys()):
                data_dict[video_name][frame_name] = global_features[index]  #reparameterize by configuration value
                continue
        #print(global_features[index])
        #print(data_dict[video_name][frame_name])
        data_dict[video_name][frame_name] = np.concatenate([global_features[index],data_dict[video_name][frame_name]], axis = 0)
          

    #print(data_dict)
   
    db_dict = {}
    # Each element is of the form 'Video<n>': [ [] ,
    #                                           [] ....,
    #                                         ]
    for video_name in data_dict.keys():
       for frame_name in data_dict[video_name].keys() :
            if(video_name not in db_dict.keys()):
                 db_dict[video_name] =data_dict[video_name][frame_name]            #reparameterize by configuration value
                 continue
            print(db_dict[video_name].shape)
            print(data_dict[video_name][frame_name].shape)

            db_dict[video_name] = np.concatenate([db_dict[video_name],data_dict[video_name][frame_name]],axis = 0)
    print(db_dict['Video2'].shape)

    #Push to new collection "Global_Features" in the database
    #client = MongoClient("mongodb://localhost:27017/")
    #mydatabase = client['InstanceRetrieval']
    #db.Global_Features.insert_many(db_dict)
    

    
def Rescale(image, width, height):
    """Rescale the proposals to a given fixed size
    """
    return cv.resize(image,(width,height))

def Crop(image,x,y,w,h) :
    """Crop the bounding box
    """
    return image[y:y+h,x:x+w]
    
    
def KMeans(np_array):
	"""
	Function to spatially reduce dimensions of features per frame
	"""
	kmeans = KMeans(n_clusters = 2 , random_state= 0).fit(np_array)
	return kmeans.cluster_centers_   ## Return the final k centroids per frame and store it in the data base as a dictionary of the form {"Centroid_1": <..> , "Centroid_2" :<..>}

def reduceDimensions(matrix):
	"""
	Function to reduce individual dimension of feature vectors extracted from NN
	"""
	pca = PCA(n_components = 3 , whiten = True)
	tfd_matrix = pca.fit_transform(matrix)
	#print(pca.explained_variance_ratio_)

	return matrix


class Dataset_Primary(Dataset):
    """
    Custom class for creating PyTorch tensors from Primary database
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
            #print("Reading document and bounding_box",proposal)
            #print(bounding_boxes[proposal])

            try:  #Some bounding boxes have inconsistent sizes....needs to be checked
               x,y,w,h=bounding_boxes[proposal][0],bounding_boxes[proposal][1],bounding_boxes[proposal][2],bounding_boxes[proposal][3]
            except Exception as e:
            	continue
            	print(str(e))

            img_path= os.path.join(document["Video"],document["Frame"])
            temp_image = cv.imread(img_path)
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



class DataSet_Secondary(Dataset) :
    """
    Custom class for creating PyTorch tensors from Primary database. <Input for the K-sparse autoencoder>
    """
    def __init__(self):
    	return True


if ( __name__ == "__main__"):
         getFeatures()