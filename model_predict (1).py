#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
import efficientnet.keras as efn 
from keras.models import load_model
from keras.preprocessing import image
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pandas as pd

# In[4]:


# Object Detection
saved_model_dir = './fashionpedia-api/model'  # specify the model dir here


session = tf.Session(graph=tf.Graph())
_ = tf.saved_model.loader.load(session, ['serve'], saved_model_dir)


# In[5]:


model = load_model(r"C:\Users\yash chaudhary\Desktop\flipkart\model.h5")
#model.compile()
model._make_predict_function()


# In[2]:


def check_t_shirt_site(path):
    
    with open(path,'rb') as f:
        np_image_string = np.array([f.read()])
    
    num_detections, detection_classes, detection_scores, image_info = session.run(['NumDetections:0','DetectionClasses:0', 'DetectionScores:0','ImageInfo:0']
                                                                                      ,feed_dict = {'Placeholder:0':np_image_string})
    num_detections = np.squeeze(num_detections.astype(np.int32), axis=(0,))
    detection_scores = np.squeeze(detection_scores, axis=(0,))[0:num_detections]
    detection_classes = np.squeeze(detection_classes.astype(np.int32), axis=(0,))[0:num_detections] 
    
    for i in range(num_detections):
        if detection_classes[i] == 2 and detection_scores[i] > 0.90 : 
            return True
    return False



# In[3]:


# returns embedding matrix for a folder of images
def generate_embedding_matrix_sites(path):
    embeddings = []
    for file in os.listdir(path):
        try:
            if 'gif' in (path + '/' + file):
                continue
            if 'png' in (path + '/' + file):
                continue
            if 'csv' in (path + '/' + file):
                continue
            print(file)
            if check_t_shirt_site(path + '\\' + file):
                
                img = image.load_img(path + '\\' + file,target_size = (300,225))
                img = image.img_to_array(img)
                img = img/255
                img = np.expand_dims(img,axis = 0)
                embedding = model.predict(img)
                embedding = np.squeeze(embedding)
                embeddings.append([path + '\\' + file,embedding])
        except:
            continue
    return np.array(embeddings)   
        


# In[6]:


ontology = json.load(open('./fashionpedia-api/data/demo/category_attributes_descriptions.json'))
def check_t_shirt(path):
    
    with open(path,'rb') as f:
        np_image_string = np.array([f.read()])
    
    num_detections, detection_boxes, detection_classes, detection_scores, image_info = session.run(['NumDetections:0','DetectionBoxes:0','DetectionClasses:0', 'DetectionScores:0','ImageInfo:0']
                                                                                      ,feed_dict = {'Placeholder:0':np_image_string})
    num_detections = np.squeeze(num_detections.astype(np.int32), axis=(0,))
    detection_boxes = np.squeeze(detection_boxes * image_info[0, 2], axis=(0,))[0:num_detections]
    detection_scores = np.squeeze(detection_scores, axis=(0,))[0:num_detections]
    detection_classes = np.squeeze(detection_classes.astype(np.int32), axis=(0,))[0:num_detections] 
    
    for i in range(num_detections):
        if detection_classes[i] == 2 and detection_scores[i] > 0.90 : 
            top,left,bottom,right = detection_boxes[i]
            bbox = (left,top,right,bottom)
            return (True,bbox)
    return (False,(0,0,0,0))    


# In[7]:


def generate_embedding_matrix_trendy(path):
    embeddings = []
    for file in os.listdir(path):
        if 'gif' in (path + '/' + file):
            continue
        if 'png' in (path + '/' + file):
            continue
        if 'csv' in (path + '/' + file):
            continue
        is_t_shirt, bbox = check_t_shirt(path + '/' + file)
        
        if is_t_shirt == False:
            continue

        img = image.load_img(path + '/' + file)
        img = img.crop(bbox)
        img = img.resize((225,300))
        img = image.img_to_array(img)
        img = img/255
        img = np.expand_dims(img,axis = 0)
        embedding = model.predict(img)
        embedding = np.squeeze(embedding)
        embeddings.append([path + '\\' + file,embedding])
    return np.array(embeddings)       
            


# In[8]:

import ast


def distance(a,b):
        return np.sqrt(np.sum((a-b)**2))
    
def find_top_neighbours(trendy_embedding_matrix,site_embedding_matrix):
    top_hits = []
    for trendy_emb in trendy_embedding_matrix:
        distances = []
        for site_emb in site_embedding_matrix:
            distances.append((site_emb[0],distance(trendy_emb[1],site_emb[1])))     
        distances = sorted(distances,key = lambda x : x[1])[:5]
        top_hits.append([trendy_emb[0],distances])

    return top_hits
        


# In[9]:


#trend = generate_embedding_matrix_trendy('/Users/mananmehta/Desktop/dataset')
#sites = generate_embedding_matrix_sites('/Users/mananmehta/Desktop/dataset 2')


# In[13]:


#hello = find_top_neighbours(trend,sites)


# In[17]:


#axes = []
#fig = plt.figure(figsize = (20,20))
#c = 0
#for i in range(len(hello)):
#    img = image.load_img(hello[i][0],target_size = (300,225))
#    axes.append(fig.add_subplot(14,6,i+c+1))
#    plt.imshow(img)
#    for j in range(len(hello[i][1])):
#        axes.append(fig.add_subplot(14,6,i+j+2+c))
#        img = image.load_img(hello[i][1][j][0],target_size = (300,225))
#        plt.imshow(img)
#    c += 5
#plt.show()
import pickle
if __name__=='__main__':
    site_folder=r'C:\Users\yash chaudhary\Desktop\flipkart\2020-08-09'
    trendy_folder=r'C:\Users\yash chaudhary\Desktop\flipkart\static\images'
 #  site_sub_folder = [ f.path for f in os.scandir(site_folder) if f.is_dir() ]
 #   trendy_sub_folder = [ f.path for f in os.scandir(trendy_folder) if f.is_dir() ]
 #   all_sites=[]
 #   all_trends=[]
 #   for site in site_sub_folder:
 #       sites=generate_embedding_matrix_sites(site+'\\images\\')
 #       all_sites.append(sites)
 #   for tre in trendy_sub_folder:
 #       trend =generate_embedding_matrix_trendy(tre+'\\')
#        all_trends.append(trend)
#    all_trends=np.array(all_trends)
#    all_sites=np.array(all_sites)
#    print(all_trends)
    site=np.load('sites_final.npy',allow_pickle=True)
    trend=np.load('trend_final.npy',allow_pickle=True)
    li=[]
    for s in site:
        for t in trend:
            temp=find_top_neighbours(t,s)
            if len(temp)==0:
                continue
            li.append(temp)
    with open("data_show.txt", "wb") as fp:
         pickle.dump(li, fp)
    #site=pd.read_csv('sites_final.csv')
    #trend=pd.read_csv('trend_final.csv')
    #print(find_top_neighbours(trend,site))
# In[ ]:




