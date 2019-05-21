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
from torchvision.models import densenet121
from torchvision.models import vgg16
from pymongo import MongoClient
import json
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import h5py
import csv

cuda = torch.cuda.is_available()
print("Is Cuda available?",cuda)
device = torch.device('cuda:0' if cuda else 'cpu')


def getFeatures():

    """
    Driver function for preprocessing(embedding extraction, PCA, KMeans)
    """
    dataSet = Dataset_Primary() 

    #Configuration dump from json config file
    filename = os.path.abspath('../config/data_config.json')
    if filename:
      with open(filename, 'r') as f:
        datastore = json.load(f)

    num_proposals = int(datastore["num_proposals"]) 
    num_clusters = int(datastore["num_clusters"])
    pca_components = datastore["pca_components"]
    data_path = datastore["data_path"]



    #Define the model
    base_model = inception_v3(pretrained=True)
    #base_model = vgg16(pretrained=True)
    #print(base_model)
    base_model = nn.Sequential(*list(base_model.children())[:-4])  #Extract features from final max pool layer
    base_model.to(device)
    #print(base_model)
    
    dataset_loader = torch.utils.data.DataLoader(dataSet, batch_size = 1 , shuffle = False, num_workers = 4)
    train_iter = iter(dataset_loader)

    feature_labels = []
    raw_features = []

    reduced_features = np.zeros(1000)   ##remove
 
    index = -1
    while(1):
      print('Memory Usage:')
      print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
      print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

      index += 1
      try:	
        image, label = train_iter.next()   #Images and labels generated according to DataSet_Primary's class definition
        print("retrieved image :","with label:", label, "of size", image.shape )
        image = Variable(image).data.cpu().numpy()
        #image= Variable(image)

      except Exception as e:
      	print("Exception :", str(e))
      	break

      #Transform the image
      #_,n,w,h,c=image.shape   # n:number of proposals , w:width, h:height, c:channels
      print("Image shape before loop:",image.shape)
      #image.resize_(n,c,h,w)
      row_count = 0

      for row in image[0]:
        row_count += 1
        #image = Variable(torch.tensor(image,device=device).float())
        print("Image  number",row_count,"with shape in loop before resizing: ",row.shape)
        w,h,c= row.shape
        row = np.expand_dims(np.reshape(row,(c,h,w)), axis = 0)
        print("Image  shape in loop after resizing: ",row.shape)
        #output_row = base_model(row.to(device))
        if row_count == 1:
            output = base_model(Variable(torch.tensor(row).float().to(device))).cpu().data.numpy()
        else:
            output = np.concatenate([output,base_model(Variable(torch.tensor(row).float().to(device))).cpu().data.numpy()], axis=0)
      print("retrieved nn features")
      #output = output.cpu().data.numpy()    #converting torch float to numpy for further processing
      print("Feature Vector shape:",output.shape)
      #print("Label :", label)
      
      raw_output = output
      for raw_feature in raw_output.tolist():
          raw_features.append(raw_feature)
      #num_rows,num_cols = output.shape
      #if num_rows < pca_components:
      #	  pca_components = num_rows

      #output = reduceDimensions(output , pca_components)                       #Call to PCA with (num of boxes * feature size) 3D matrix as input
      #print("Shape out output after PCA:" , output.shape)
      num_rows,_ = output.shape
      if num_rows > num_clusters :
        output = reduceCentres(output,num_clusters)                              #Call to K-Means with cluster centres derived from config file
      else :
        continue	
      print("Shape of output after Kmeans", output.shape)

      #Append features to create an M*1000 matrix for all M proposals
      if index  == 0 :
      	reduced_features = output
      	feature_labels.append(label[0])
      	continue

      reduced_features = np.concatenate([reduced_features,output], axis = 0)
      print("Extracted bounding box data from Frame db for: ",label[0])
      feature_labels.append(label[0])


    print("reduced Features size ",reduced_features.shape)
    print("reduced Feature labels length",len(feature_labels))
    
    reduced_dict = {}
    #Each element is of the form 'Video1':{  {'Frame1':np.array of size 1000, 
    #                                                           'Frame2':np.array....
    #                                                           }
    #                                                        }
                          

    #print(type(global_labels[0]))
    

    #Push to new collection "Reduced_Features" and "Raw_Features" in the database
    client = MongoClient("mongodb://localhost:27017/")
    mydatabase = client['InstanceRetrieval']
    Raw_Features = mydatabase["Raw_Features"]
    Reduced_Features = mydatabase["Reduced_Features"]   #Database that is indexed by "Video"+"Frame" keys and contains features with shape (num of proposals)*(reduced proposal feature size)
    
    #Call to PCA to reduce number of columns of dataset to num_proposals stored in config file
    
    reduced_features = reduceDimensions(reduced_features,pca_components)


    for index, _ in enumerate(feature_labels):
        video_name = feature_labels[index].split("/")[0]
        frame_name = feature_labels[index].split("/")[1][:-4]
        if(video_name not in reduced_dict.keys()):
        	reduced_dict[video_name] = {}
        print("Dictionary element shape not in list form", reduced_features[index*num_clusters:index*num_clusters+num_clusters,:].shape)
        reduced_dict[video_name][frame_name] = reduced_features[index*num_clusters:index*num_clusters+num_clusters,:].tolist()   #Each element is a list of lists
        mydatabase.Reduced_Features.insert( {"_id": video_name+"/"+frame_name, "item":reduced_dict[video_name][frame_name]})

    f = open("Features.csv","w")
    writer = csv.DictWriter(f,fieldnames = ["id","feature"])
    feature_count = 0
    for raw_feature in raw_features:
        feature_count += 1
        writer.writerow({"id":feature_count,"feature":raw_feature})
        #mydatabase.Raw_Features.insert({"_id": feature_count , "item": raw_features})

    
def Rescale(image, width, height):
    """Rescale the proposals to a given fixed size
    """
    return cv.resize(image,(width,height))

def Crop(image,x,y,w,h) :
    """Crop the bounding box
    """
    return image[y:y+h,x:x+w]
    
    
def reduceCentres(np_array, num_clusters):
	"""
	Function to spatially reduce dimensions of features per frame
	"""
	kmeans = KMeans(init='random', n_clusters= num_clusters).fit(np_array)
	return kmeans.cluster_centers_   ## Return the final k centroids per frame by retaining dimension length

def reduceDimensions(matrix, n_components):
    """
    Function to reduce individual dimension of feature vectors extracted from NN
    """
    pca = PCA( whiten = True, n_components = n_components)
    matrix = pca.fit_transform(matrix)
    print("Shape of reduced matrix is:", matrix.shape)
    return matrix


class Dataset_Primary(Dataset):
    """
    Custom class for creating PyTorch tensors from Primary database
    """
    def __init__(self, transform=None):
        if os.path.exists("dataset.h5"):
            os.remove("dataset.h5")
        hdf5_file = h5py.File("dataset.h5",'w')
        

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
        
        n=-1

        db_iterable = collection.find()
        for document in db_iterable:
          n+=1
          img_path= os.path.join(document["Video"],document["Frame"])
          print("Retrieving proposals for Video:", document["Video"],"and Frame: ", document["Frame"])
          bounding_boxes = document["Bounding_Boxes"]	
          frame_features=[]
          for proposal in bounding_boxes: #To iterate through all proposals in a given frame
            
            print("Reading document and bounding_box",proposal)
            #print(bounding_boxes[proposal])

            try:  #Some bounding boxes have inconsistent sizes....needs to be checked
               x,y,w,h=bounding_boxes[proposal][0],bounding_boxes[proposal][1],bounding_boxes[proposal][2],bounding_boxes[proposal][3]
            except Exception as e:
                print(str(e))
                continue

            
            temp_image = cv.imread(img_path)
            temp_image = Crop(temp_image,x,y,w,h)
            box_features = Rescale(temp_image,128,128)
            frame_features.append(box_features)
         
          hdf5_file.create_dataset(str(n), data= np.array(frame_features))   #image matrix constists of all bounding boxes per frame stacked vertically indexed by frame number
          labels += [os.path.join(str(img_path.split("/")[-1:-3:-1][1]) , str(img_path.split("/")[-1:-3:-1][0]))] #label is of the form Video_X/Frame_Y
          

        hdf5_file.close()
        #self.images = torch.from_numpy(images).float()
        self.labels = labels

        
    def __getitem__(self, index):
        """
        Retrieve frame level features of size (num_of_proposals * feature size of proposal) 

        """
        hdf5_file = h5py.File("dataset.h5",'r')
        hdf5_data  = hdf5_file[str(index)].value
        #image = torch.from_numpy(hdf5_data).float()
        image = hdf5_data
        hdf5_file.close()
        #image = self.images[index]
        frame = self.labels[index]
        
        #print("returning image from hdf5:", image ,"with label", frame)
        return image, frame
    
    def __len__(self):
        return len(self.labels)


if ( __name__ == "__main__"):

        getFeatures()