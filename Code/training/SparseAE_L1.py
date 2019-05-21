import warnings
warnings.filterwarnings('ignore')
import os, datetime
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torchvision
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.utils.data as utils
from torch.autograd import Variable
from torchvision.models.inception import inception_v3
from pymongo import MongoClient
from sklearn.decomposition import PCA
import numpy as np
import json
import re
import csv
import cv2 as cv
import ast


cuda = torch.cuda.is_available()

print("Is Cuda available?",cuda)
device = torch.device('cuda:0' if cuda else 'cpu')

class SparseAutoencoderL1(nn.Module):
    def __init__(self):
        super(SparseAutoencoderL1, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 100),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class DataSet_Secondary(Dataset) :
    """
    Custom class for creating PyTorch tensors from Primary database for each video. <Input for the K-sparse autoencoder>
    """
    def __init__(self,cur_video,transform = None):
         #Returns all labels and frame level features for the input video 
         match_string = "^"+cur_video+"/"
         regx = re.compile(match_string, re.IGNORECASE)
         
         #Creating datasets and data loaders
         client = MongoClient("mongodb://localhost:27017/")
         mydatabase = client['InstanceRetrieval']
         collection = mydatabase['Reduced_Features']
         db_iterable = collection.find({"_id":regx})
         
         features = []
         labels = []    

         for document in db_iterable:
                    video,frame = document["_id"].split("/")
                    features.append(np.asarray(document["item"]))
                    labels.append(frame)

         self.labels = labels 
         self.features = features

    def __getitem__(self, index):
        """
        Retrieve indexed frame level features for the input video

        """
        #print("Print retrieved image for label:", self.labels[index])

        feature = self.features[index]
        label = self.labels[index]

        
        return feature, label
    
    def __len__(self):
        return len(self.labels)       


def sparse_loss(autoencoder, images):
    loss = 0
    values = images
    for i in range(3):
        fc_layer = list(autoencoder.encoder.children())[2 * i]
        relu = list(autoencoder.encoder.children())[2 * i + 1]
        values = relu(fc_layer(values))
        loss += torch.mean(torch.abs(values))
    for i in range(2):
        fc_layer = list(autoencoder.decoder.children())[2 * i]
        relu = list(autoencoder.decoder.children())[2 * i + 1]
        values = relu(fc_layer(values))
        loss += torch.mean(torch.abs(values))
    return loss



def model_training(autoencoder, train_dataset, epoch, video, BATCH_SIZE):
    #Note: Batch size should be equal to the number of centres for Kmeans clustering 
    loss_metric = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE , shuffle = False, num_workers = 4)
    

    autoencoder.train()  

    train_iter = iter(train_loader)
    i=0
    while(1):  #While loop to iterate over frames of the current video
      #try:
        i+=1
        optimizer.zero_grad()
        try:
          image_array, labels = train_iter.next()
          print("Shape of retrieved image array for label",labels," is :" , image_array.shape)
        except Exception as e:
            print("Exception",str(e))
            break
        print(image_array.size())
        for images in image_array:  #each iterated image is a compact proposal of a frame in the video(after Kmeans and PCA)
            print("i=",i)
            print(images.shape)
            images = Variable(images).float()
            #images = images.view(images.size(0), -1).float()
            if cuda: images = images.to(device) #loading tensors to GPU if available
            outputs = autoencoder.forward(images)
            mse_loss = loss_metric(outputs, images)
            l1_loss = sparse_loss(autoencoder, images)
            loss = mse_loss + SPARSE_REG * l1_loss
            loss.backward()
            optimizer.step()
            if (i + 1) % LOG_INTERVAL == 0:
               print('Epoch [{}/{}] - Iter[{}/{}], Total loss:{:.4f}, MSE loss:{:.4f}, Sparse loss:{:.4f}'.format(
                    epoch + 1, EPOCHS, i + 1, len(train_loader.dataset) // BATCH_SIZE, loss.item(), mse_loss.item(), l1_loss.item()
                ))
    
def evaluation(autoencoder, test_dataset, video, BATCH_SIZE=1):
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = BATCH_SIZE , shuffle = False, num_workers = 4)
    test_iter = iter(test_loader)

    total_loss = 0
    loss_metric = nn.MSELoss()
    autoencoder.eval()

    while(1):
       try: 

          image_array, labels = test_iter.next()
       except Exception as e:
          print(str(e))
          break    
      
       for images in image_array:
         images = Variable(images).float()
         #images = images.view(images.size(0), -1)
         if cuda: images = images.to(device).float()
         outputs = autoencoder.forward(images)
         loss = loss_metric(outputs, images)
         total_loss += loss * len(images)
         avg_loss = total_loss / len(test_loader.dataset)
         print('\nAverage MSE Loss on Test set: {:.4f}'.format(avg_loss))
         global BEST_VAL
         if TRAIN_SCRATCH and avg_loss < BEST_VAL:
           BEST_VAL = avg_loss
           torch.save(autoencoder.state_dict(), './history/'+video+'_sparse_autoencoder_l1.pt')
           print('Save Best Model in HISTORY\n')


def retrieve_videos(model_path, query_path, query_name ,k ):
    """
    Function to retrieve top 'k' Videos with the lowest query reconstruction error
    """
    loss_metric = nn.MSELoss()
    autoencoder.eval()
    query=query_name
    match_string = "^"+query
    regx = re.compile(match_string, re.IGNORECASE)
    client = MongoClient("mongodb://localhost:27017/")
    mydatabase = client['InstanceRetrieval']
    collection = mydatabase['Query_Features']
    query_object = collection.find({"query":regx})
    for feature in query_object:
      if not feature:
          print("Unknown query")  
      query_array = np.array(feature["feature"])

    video_dict = {}

    for model in os.listdir(model_path):
        autoencoder.load_state_dict(torch.load(os.path.join(model_path,model))) 
        outputs = autoencoder.forward(Variable(torch.tensor(query_array, device=device).float()))
        loss = loss_metric(Variable(torch.tensor(query_array).double()),outputs.cpu().double())
        video_dict[model.split("_")[0]] = loss

    sorted_dict= sorted(video_dict.items(),key= lambda x: x[1])
    return [ item[0] for item in sorted_dict[0:k]]

def normalize_query(query_path):
    #Creating datasets and data loaders
    client = MongoClient("mongodb://localhost:27017/")
    mydatabase = client['InstanceRetrieval']
    #collection = mydatabase['Raw_Features']
    #db_iterable = collection.find()

    
    #Loop to extract all feature vectors from training phase
    #for document in db_iterable:
        
    #       feature_matrix=  np.array(feature_list)
    
    feature_list = []
    with open(os.path.join("../","data-handler","Features.csv") ) as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            row_list = ast.literal_eval(row[1])
            feature_list.append(row_list)

    feature_matrix = np.array(feature_list[:3000]) #Taking a small chunk of the entire dataset to prevent memory overflow
    print(feature_matrix.shape)
    pca = PCA( whiten = True, n_components = pca_components)
    pca.fit(feature_matrix)

    #Iterate through query dataset and extract features
    for query in os.listdir(query_path):
         query_name = query
         query = cv.imread(os.path.join(query_path,query))
         query = cv.resize(query,(128,128))
         print("query shape:",query.shape)
         w,h,c=query.shape   # n:number of proposals , w:width, h:height, c:channels
         query=query.reshape(c,h,w)
         query = query[:,:,:,np.newaxis]    
         query=query.reshape(1,c,h,w)
         print("new query shape:",query.shape)
         query = torch.from_numpy(query) #converting numpy to torch tensor
         query = Variable(query)         #wrapping a torch tensor as a variable
         #query = query.view(query.size(0), -1) #resizing variable
         query = query.cuda()
         video_dict= {} #The list of videos sorted according to score
         autoencoder = SparseAutoencoderL1()
         autoencoder.eval()
         autoencoder = autoencoder.to(device)
         loss_metric = nn.MSELoss()

   
         #Define the model
         base_model = inception_v3(pretrained=True)
         #print(base_model)
         base_model = nn.Sequential(*list(base_model.children())[:-4])  #Extract features from final max pool layer
         #print(base_model)
         if cuda: 
            query = query.to(device).float()
            base_model = base_model.to(device)

         query_output = base_model(query)   # returns 1000 dimensional vector
         query_output = query_output.cpu().data.numpy()
         query_output = pca.transform(query_output)
           
         print("query output shape:",query_output.shape)
         

         mydatabase.Query_Features.insert({"query":query_name.split(".")[0],"feature":query_output.tolist()})
    


if __name__ == '__main__':


    #Configuration dump from json config file
    filename = os.path.abspath('../config/data_config.json')
    if filename:
      with open(filename, 'r') as f:
        datastore = json.load(f)


    #Retrieving configuration parameters
    data_path = datastore["data_path"]
    query_path = datastore["query_path"]
    model_path= datastore["model_path"]
    search_k = int(datastore["search_k"])
    num_proposals = int(datastore["num_proposals"]) 
    num_clusters = int(datastore["num_clusters"])
    pca_components = int(datastore["pca_components"])
    EPOCHS = int(datastore["EPOCHS"])
    BATCH_SIZE = int(datastore["BATCH_SIZE"])
    LEARNING_RATE = float(datastore["LEARNING_RATE"])
    WEIGHT_DECAY = float(datastore["WEIGHT_DECAY"])
    LOG_INTERVAL = int(datastore["LOG_INTERVAL"])
    SPARSE_REG = float(datastore["SPARSE_REG"])
    TRAIN_SCRATCH = datastore["TRAIN_SCRATCH"]        # whether to train a model from scratch
    SEARCH_PHASE = datastore["SEARCH_PHASE"]
    BEST_VAL = datastore["BEST_VAL"]     # record the best val loss
    if BEST_VAL == "inf":
        BEST_VAL = float('inf')
    
    autoencoder = SparseAutoencoderL1()
    if cuda: autoencoder.to(device)


    num_videos = 0
    if  SEARCH_PHASE == "False":
      for video in os.listdir(data_path):
        num_videos +=1  



       
        
        if TRAIN_SCRATCH == "True":
            dataset = DataSet_Secondary(video)
            train_dataset = dataset
            test_dataset = dataset
            # Training autoencoder from scratch
            print("Training model for video :", video)
            for epoch in range(EPOCHS):
                starttime = datetime.datetime.now()
                model_training(autoencoder, train_dataset, epoch, video, BATCH_SIZE)
                endtime = datetime.datetime.now()
                print(f'Train an epoch in {(endtime - starttime).seconds} seconds')
                # evaluate on test set and save best model
                print("Evaluating model for video :", video)
                evaluation(autoencoder, test_dataset, video, BATCH_SIZE)
            print('Training Complete with best validation loss {:.4f}'.format(BEST_VAL))

        else:
            print("Entering Validation phase")
           # autoencoder.load_state_dict(torch.load('./history/'+video+'sparse_autoencoder_l1.pt'))
           # evaluation(autoencoder, test_dataset, video, BATCH_SIZE)

           # autoencoder.cpu()
           # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle = False, num_workers = 4)
           # dataiter = iter(test_loader)
           # images, _ = next(dataiter)
           # images = Variable(images)
           # outputs = autoencoder(images.view(images.size(0), -1))
    else :
      #Search phase
      print("Entered Search Phase")
      normalize_query(query_path) #To rescale Query features to match dimensions of input features
      print("Normalized Queries")
      query_name = "Query7"
      print("The top ", search_k, "retrieved videos are: " ,retrieve_videos(model_path,query_path,query_name,search_k))  #query is the name of the query image
      







