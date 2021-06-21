import face_recognition
import cv2
import numpy as np
import os
import glob
import re
import pickle 
import threading
from datetime import datetime
from pymongo import MongoClient
import time
from tkinter import *
from PIL import ImageTk, Image
import cv2
import datetime
import time
import configparser
from gtts import gTTS
from playsound import playsound


frame_name='%H_%M_%S_%d_%m_%Y.jpg'

config = configparser.ConfigParser()
config.read('config.ini')

def isaccepted():
   print("yes")

try: 
	conn = MongoClient() 
	print("Connected successfully!!!") 
except: 
	print("Could not connect to MongoDB") 


# database 
db = conn.face 
collection = db.face
cur=collection.find()

known_face_names=[]
known_face_encodings=[]

# fetch all the encodings and names from database
for i in list(cur):
    known_face_names.append(i['key'])
    known_face_encodings.append(i['encodings'])







def open_details_frame(name):
    qwerty=collection.find_one({'key':name})
process_this_frame = True
temp_face_encodes=[] 


root = Tk()
# Create a frame
app = Frame(root, bg="white")
button1=Button(app,text="",command=isaccepted)
app.grid()
button1.grid()
# Create a label in the frame
lmain = Label(app)
lmain.grid()

# Capture from camera
video_capture = cv2.VideoCapture(0)

temp_encodings=[]
temp_time=datetime.datetime.now()
old_face_encodings=[]


def video_stream(frame,name):
        # _, frame = cap.read()
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        #mycursor = mydb.cursor()
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(name+datetime.datetime.now().strftime(frame_name))
        fname=config['test']['save_images']+"/"+name+datetime.datetime.now().strftime(frame_name)
        print(fname)
        cv2.imwrite(fname, frame)

    
while True:
    process_this_frame = True
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
             
            
        face_names = []
    
        name=" blank"
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding,tolerance=0.46)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            print(face_distances)
            print(np.argmin(face_distances))
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]


            face_names.append(name)



	    
            
    process_this_frame = not process_this_frame


    # Display the results
    
    # temp_encodings=frame
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        # if len(face_encodings) == 1:
          # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
          
          # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        face_names=list(dict.fromkeys(face_names)) 

        if '_'.join(face_names)=='Unknown':
            myobj = gTTS(text="Unknown Person Detected", lang="en", slow=False)
            myobj.save("temp.mp3")
            playsound('temp.mp3')
            button1['text']=name
            button1['bg']="red"
            button1['command']=lambda: open_details_frame(name)
        else:
            myobj = gTTS(text=name+" is Detected", lang="en", slow=False)
            myobj.save("temp.mp3")
            playsound('temp.mp3')
            button1['text']='_'.join(face_names)
            button1['bg']="red"
            button1['command']=lambda: donothing()


                
    name='_'.join(face_names)    
    

    video_stream(frame,name)
    temp=name
    temptime=datetime.datetime.now()
    temp_encodings=frame 
    root.update()
   
 
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()
   

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()    
   
