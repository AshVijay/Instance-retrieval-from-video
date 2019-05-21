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
from pymongo import MongoClient
import numpy as np
import json



cuda = torch.cuda.is_available()

print("Is Cuda available?",cuda)
device = torch.device('cuda:0' if cuda else 'cpu')

class SparseAutoencoderL1(nn.Module):
    def __init__(self):
        super(SparseAutoencoderL1, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(10000, 128),
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
            nn.Linear(128, 10000),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class DataSet_Secondary(Dataset) :
    """
    Custom class for creating PyTorch tensors from Primary database. <Input for the K-sparse autoencoder>
    """
    def __init__(self,video,transform = None):
         #Creating datasets and data loaders
         client = MongoClient("mongodb://localhost:27017/")
         mydatabase = client['InstanceRetrieval']
         collection = mydatabase['Global_Features']
         db_iterable = collection.find()
         
         features = []
         labels = []    

         for document in db_iterable:
            for key in document.keys():
              if key == "_id":
                continue
              if key == video:
                for frame in document[video].keys():
                  features.append(np.asarray(document[video][frame]))
                  labels.append(frame)

         self.labels = labels 
         self.features = features

    def __getitem__(self, index):
        """
        Retrieve indexed indexed image

        """

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


def recon_loss(video, model ,query):
    """Function to implement the reconstruction loss for each query-video pair"""
    return True


def model_training(autoencoder, train_dataset, epoch, BATCH_SIZE):
    loss_metric = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE , shuffle = False, num_workers = 4)
    

    autoencoder.train()  

    train_iter = iter(train_loader)
    i=0
    while(1):
      try:
        i+=1
        optimizer.zero_grad()
        images, labels = train_iter.next()
        #Display figures 
        plt.imshow(images)
        plt.show(block=False)
        plt.pause(1)
        plt.close('all')
        
        print("label",labels, "i=",i)
        #images = data
        images = Variable(images)
        images = images.view(images.size(0), -1).float()
        if cuda: images = images.to(device)
        outputs = autoencoder(images.float())
        mse_loss = loss_metric(outputs, images)
        l1_loss = sparse_loss(autoencoder, images)
        loss = mse_loss + SPARSE_REG * l1_loss
        loss.backward()
        optimizer.step()
        if (i + 1) % LOG_INTERVAL == 0:
            print('Epoch [{}/{}] - Iter[{}/{}], Total loss:{:.4f}, MSE loss:{:.4f}, Sparse loss:{:.4f}'.format(
                epoch + 1, EPOCHS, i + 1, len(train_loader.dataset) // BATCH_SIZE, loss.item(), mse_loss.item(), l1_loss.item()
            ))
      except Exception as e:
          print("Training interrupted by the following exception : ", str(e))
          break

def evaluation(autoencoder, test_dataset, BATCH_SIZE):
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = BATCH_SIZE , shuffle = False, num_workers = 4)
    test_iter = iter(test_loader)

    total_loss = 0
    loss_metric = nn.MSELoss()
    autoencoder.eval()
    while(1):
       try: 

          images,_ = test_iter.next()
          images = Variable(images)
          images = images.view(images.size(0), -1)
          if cuda: images = images.to(device).float()
          outputs = autoencoder(images)
          loss = loss_metric(outputs, images)
          total_loss += loss * len(images)
       except Exception as e:
          print("The following exception occured (ignoring):", str(e))   
          break

    avg_loss = total_loss / len(test_loader.dataset)

    print('\nAverage MSE Loss on Test set: {:.4f}'.format(avg_loss))

    global BEST_VAL
    if TRAIN_SCRATCH and avg_loss < BEST_VAL:
        BEST_VAL = avg_loss
        torch.save(autoencoder.state_dict(), './history/sparse_autoencoder_l1.pt')
        print('Save Best Model in HISTORY\n')


def retrieve_videos(autoencoder, test_dataset, ):
    """
    Function to retrieve the Video with the lowest query reconstruction error
    """
    



if __name__ == '__main__':


    #Configuration dump from json config file
    filename = os.path.abspath('../data-handler/data_config.json')
    if filename:
      with open(filename, 'r') as f:
        datastore = json.load(f)

    data_path = datastore["data_path"]




    EPOCHS = 5
    BATCH_SIZE = 1
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    LOG_INTERVAL = 100
    SPARSE_REG = 1e-3
    TRAIN_SCRATCH = True        # whether to train a model from scratch
    BEST_VAL = float('inf')     # record the best val loss

    
    for video in os.listdir(data_path):
          
        dataset = DataSet_Secondary(video)
        train_dataset = dataset
        test_dataset = dataset


        autoencoder = SparseAutoencoderL1()
        if cuda: autoencoder.to(device)

        if TRAIN_SCRATCH:
            # Training autoencoder from scratch
            print("Training model for video :", video)
            for epoch in range(EPOCHS):
                starttime = datetime.datetime.now()
                model_training(autoencoder, train_dataset, epoch, BATCH_SIZE)
                endtime = datetime.datetime.now()
                print(f'Train a epoch in {(endtime - starttime).seconds} seconds')
                # evaluate on test set and save best model
                print("Evaluating model for video :", video)
                evaluation(autoencoder, test_dataset, BATCH_SIZE)
            print('Trainig Complete with best validation loss {:.4f}'.format(BEST_VAL))

        else:
            autoencoder.load_state_dict(torch.load('./history/sparse_autoencoder_l1.pt'))
            evaluation(autoencoder, test_loader)

            autoencoder.cpu()
            dataiter = iter(train_loader)
            images, _ = next(dataiter)
            images = Variable(images[:32])
            outputs = autoencoder(images.view(images.size(0), -1))

            # plot and save original and reconstruction images for comparisons
            plt.figure()
            plt.subplot(121)
            plt.title('Original MNIST Images')
            data_utils.imshow(torchvision.utils.make_grid(images))
            plt.subplot(122)
            plt.title('Autoencoder Reconstruction')
            data_utils.imshow(torchvision.utils.make_grid(
                outputs.view(images.size(0), 1, 28, 28).data
            ))
            plt.savefig('./images/sparse_autoencoder_l1.png')