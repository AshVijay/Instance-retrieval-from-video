#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This sample demonstrates structured edge detection and edgeboxes.
Usage:
  edgeboxes_demo.py [<model>] [<input_image>]
'''

import cv2 as cv
import numpy as np
import sys
import os
from pymongo import MongoClient
import json
import matplotlib.pyplot as plt



   
class dataHandler:
	"""
	Class that handles object proposal and data transfer operations
	"""


	def generate_proposals(self,video_path,image_path,model,num_proposals, db):
		
		im = cv.imread(os.path.join(video_path,image_path))
		print("Generating proposals for ", os.path.join(video_path,image_path))
		#Piotr and Dollar(2014) Edge box implementation using OpenCV
		edge_detection = cv.ximgproc.createStructuredEdgeDetection(model)
		rgb_im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
		edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)
		orimap = edge_detection.computeOrientation(edges)
		edges = edge_detection.edgesNms(edges, orimap)
		edge_boxes = cv.ximgproc.createEdgeBoxes()
		edge_boxes.setMaxBoxes(num_proposals)
		boxes = edge_boxes.getBoundingBoxes(edges, orimap)
		print("Retrieved boxes",boxes,"For Frame",image_path)
		
		box_count= 0
		box_dict={}
		for b in boxes:
			box_count  += 1 
			x, y, w, h = b
			print(type(x))
			cv.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 1, cv.LINE_AA)
			box_dict['Box'+str(box_count)] = [int(x),int(y),int(w),int(h)]

		#print(box_dict)
		#push to database
		db.Frame.insert_one({'Video':video_path,'Frame':image_path , "Bounding_Boxes":box_dict})

		#Display proposals
		#cv.imshow("edges", edges)
		#cv.imshow("edgeboxes", im)
		#cv.waitKey(1000)
		#cv.destroyAllWindows() 
		
		#plt.imshow(im)
		#plt.show()




if __name__ == '__main__':
	    print(__doc__)

	    #loading parameters from config file
	    with open('../config/data_config.json') as config_file:
	       config_data = json.load(config_file)
	   
	    data_path = os.path.abspath(config_data['data_path'])
	    model = config_data['rf_model']
	    num_proposals = int(config_data['num_proposals'])   
	   
	    #creating an object to handle proposal operations and data transfer
	    data_object = dataHandler()

	    # Setting up mongodb connection 
	    client = MongoClient("mongodb://localhost:27017/") 
	    mydatabase = client['InstanceRetrieval']
	    

	    for video_path in os.listdir(data_path):
	      for image_path in os.listdir(os.path.join(data_path,video_path)):
	         if(image_path.endswith(('jpg','png')) ) :  
	         	
	         	data_object.generate_proposals(os.path.join(data_path,video_path),image_path,model,num_proposals,mydatabase)



    
 
