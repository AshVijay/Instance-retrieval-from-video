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



   
class dataHandler:
	"""
	Class that handles object proposals
	"""

	def generate_proposals(self,image_path,model,num_proposals):
		im = cv.imread(image_path)

		edge_detection = cv.ximgproc.createStructuredEdgeDetection(model)
		rgb_im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
		edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)
		orimap = edge_detection.computeOrientation(edges)
		edges = edge_detection.edgesNms(edges, orimap)
		edge_boxes = cv.ximgproc.createEdgeBoxes()
		edge_boxes.setMaxBoxes(num_proposals)
		boxes = edge_boxes.getBoundingBoxes(edges, orimap)
		for b in boxes:
			x, y, w, h = b
			cv.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 1, cv.LINE_AA)
		cv.imshow("edges", edges)
		cv.imshow("edgeboxes", im)
		cv.waitKey(5000)
		cv.destroyAllWindows() 




if __name__ == '__main__':
    print(__doc__)
    #loading parameters from config file
    with open('data_config.json') as config_file:
       config_data = json.load(config_file)
    print(config_data)

    data_object = dataHandler()

    # Setting up mongodb connection 
    client = MongoClient("mongodb://localhost:27017/") 
    mydatabase = client['admin']

    
    data_path = os.path.abspath(config_data['data_path'])
    model = config_data['model']
    num_proposals = int(config_data['num_proposals'])

    for video_path in os.listdir(data_path):
      for image_path in os.listdir(os.path.join(data_path,video_path)):
         if(image_path.endswith(('jpg','png')) ) :  
            data_object.generate_proposals(os.path.join(data_path,video_path,image_path),model,num_proposals)



    
 
