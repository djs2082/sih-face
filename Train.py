#import all the required libraries
import face_recognition
import cv2
import numpy as np
import os
import glob
import re
import pickle
import sys
import logging
from pymongo import MongoClient 
from datetime import datetime
import shutil
import configparser

# Read The Config File
config = configparser.ConfigParser()
config.read('config.ini')

#create log file
log_file=config['train']['log']+'/logfile_%H_%M_%S_%d_%m_%Y.log'
logging.basicConfig(filename=datetime.now().strftime(log_file), level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S',)


#connect to MongoDB database
try: 
	conn = MongoClient() 
	logging.info('MongoDB database connected successfully')
except: 
	logging.error('Error Connecting to MongoDB database') 
db = conn.face  
collection = db.face

##########################3
def remove(list): 
    pattern = '[0-9]'
    list = [re.sub(pattern, '', i) for i in list] 
    return list
#################################3



#declare the required lists
paths=[]
list_of_files_images=[]

#get all the folders in known_people directoy
c=0
dirname = os.path.dirname(__file__)
print(dirname)
for x in os.walk(config['train']['images_to_train']):
    print(x)
    if c==0:
        c=1
        names_images=x[1]
        continue
    paths.append(os.path.join(dirname,x[0]+'/'))   
c=0
logging.info('collected all folders in known_people folder')
logging.debug(names_images)


#get all the image files in known_people
for x in paths:
    print(x)
    list_of_files_images.append([f for f in glob.glob(paths[c]+'*.jpeg')])
    c=c+1


logging.info('collected all images from all folders in known_people folder')
logging.debug(list_of_files_images)
 
#make array of sample pictures with encodings
known_face_encodings = []
known_face_names = []
face_dictionary={}

#copy list_of_files_images to list_of_files
list_of_files_images_new=list_of_files_images.copy()

#declare required variables and lists
sum=0
known_face_encodings=[]
d=0
k=0
c=0
len_x=0

#check if folder is empty
for x in list_of_files_images_new:
    print("training ",names_images[d])
    if len(x)==0:
      print(names_images[d]," folder is empty")
      logging.debug(names_images[d])
      logging.error('above folder is empty with no images')
      logging.info('exiting the program with no updates in training')
      sys.exit()


#get encodings of all images in that folder
    logging.info('started loading images and finding their encodings')
    for y in x:
        print("encoding ",y)
        logging.info('loading image')
        known_image=face_recognition.load_image_file(y)
        logging.debug(y)
        logging.info('loading image done')
        logging.info('finding the encoding of image')
        known_image_encodes=face_recognition.face_encodings(known_image)
        if len(known_image_encodes)==0:
          logging.error('no face is detected in image')
          logging.info('skiping to find encodes of this image')
          c=c+1
          print("No face detected in the photo ",y," so skipping it from training")
          len_x=len_x+1
          continue
        known_face_encodings.append(known_image_encodes[0])
        logging.info('finding encoding of image is done')
        c=c+1


#declare required variables to find average
    known_face=[]
    i=0
    j=0
    d=0

#find average of all the encodings of images
    logging.info('getting average of encodings of loaded images')
    while i<128:
        while j<len(x)-len_x:
            sum=sum+known_face_encodings[j][i]
            j=j+1
        known_face.append(float(sum/(len(x)-len_x)))   
        j=0
        sum=0
        i=i+1
    logging.info('done with getting average encodings')
    logging.info('storing the name and encoding in local dictionary')
#store the encoding in local dictionary
    face_dictionary.update({names_images[d]:known_face})
    logging.info('stored name and encodes in local dictionary')
    logging.info('storing the name and encodings in mongodb')
#store the encodings in MongoDB
    rec_id1 = collection.insert_one({'key': names_images[d], 'encodings': known_face})
    logging.info('stored all name and encodes in mongodb')
#override all the temp variables
    known_face=[]    
    known_face_encodings=[]   
#move trained images to another folder
    
    shutil.move(config['train']['images_to_train']+'/'+names_images[d].strip(),config['train']['trained_images'])
    d=d+1
    

   
