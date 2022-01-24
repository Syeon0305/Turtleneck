#!/usr/bin/env python
# coding: utf-8

# ### 1.라이브러리

# In[2]:


import time
import sys

# OpenCV
import cv2
## face
import dlib
# dataframe
import numpy as np
import pandas as pd

# Arduino 연동
import serial
   
# 경고 제어    
import warnings
warnings.filterwarnings("ignore")

# SVM 학습
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split  

# 학습된 모델 save 및 load
from sklearn import datasets
import pickle
from sklearn.externals import joblib


# ### 2. 학습을 위한 Dataset

# In[4]:


"""# final dataset 넣기
final_low_ = pd.read_csv("final_low.csv")
final_low = final_low_.iloc[:,[0,8,16,36]]

X_train_low = final_low.drop('class',axis=1)
y_train_low = final_low['class']

# final dataset으로 얻은 best parameter 넣기
svm_Detection= SVC(kernel='rbf', C=1, gamma=0.01).fit(X_train_low, y_train_low)"""


# ### 2'. pkl 파일로 학습 데이터 로드

# In[5]:


svm_from_joblib = joblib.load('1003_svm_train.pkl')


# ### 3. Detection 
# #### 1) Low 일 때

# In[9]:


# Global 변수 Setting
mode = 'A'
##거북목 판단 결과를 누적해서 담아놓은 list
X_length_low = []
X_length_high = []
y_vec = []
y_vec2 = []

def detection_low():
     
  #---------------------------------Setting-----------------------------------#
    
    # Face Detection : 1번, 9번, 17번
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    # Pose Detection : 어깨 2번, 5번
    MODE = "MPI"
    
    ## MPI Keypoints
    if MODE is "MPI" :
        protoFile = "pose_deploy_linevec_faster_4_stages.prototxt"
        weightsFile = "pose_iter_160000.caffemodel"
        nPoints = 7

    ## network 불러오기
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    threshold = 0.1

    # 얼굴과 어깨가 화면에 보이는가?
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)           # [WARN:0] global cap Error
    time.sleep(1)
 
  #-----------------------------Webcam 실행---------------------------------#
    
    ## Webcam이 실행하지 않는 경우 → 종료
    if not cap.isOpened():
        exit()
    
    ## Webcam이 실행하는 경우 → 판별 알고리즘
    while(y_vec.count(1) < 1):
        
        # 변수 Setting    
        ## Empty list to store the detected keypoints
        shoulder_points = [] 
        shoulder = []
        face_points = []
        
        ### OpenCV로 촬영되는 frame
        hasFrame, image = cap.read()
        cv2.imshow('image', image)
        frameCopy = np.copy(image)
        faces = detector(image)

        if not hasFrame:
            break
 
  #-------------------------------OpenPose-------------------------------# 
    
        ## 불러온 이미지에서 height 얻기
        frameHeight = image.shape[0]
        
        ## network에 넣기 위한 전처리
        inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255,
                                        (368, 368),(0, 0, 0), 
                                        swapRB=False, crop=False)
        ## network에 넣어주기
        net.setInput(inpBlob)
        
        ## 결과 받아오기
        output = net.forward()
        
        ## output shape 정하기
        H = output.shape[2]
        
        # Pose Detection : shoulder
        for i in range(nPoints):

            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            # Scale the point to fit on the original image
            y = (frameHeight * point[1]) / H


            if prob > threshold:
                shoulder_points.append((int(y)))
            else:
                shoulder_points.append(None)
                    
        # Pose Detection 어깨 2번, 5번의 y좌표 평균 → mean_shoulder
        ## 2번, 5번 점 detect 여부
        if shoulder_points[2] != None:
            ## (2번, 5번) = (O, O) → 두 점의 평균
            if  shoulder_points[5] != None:
                mean_shoulder = (shoulder_points[2] + shoulder_points[5])/2
            ## (2번, 5번) = (O, X) → 2번 점
            elif shoulder_points[5] == None:
                mean_shoulder = shoulder_points[2]
            else:
                pass
        
        elif shoulder_points[2] == None:
            ## (2번, 5번) = (X, O) → 5번 점
            if  shoulder_points[5] != None:
                mean_shoulder = shoulder_points[5]

            ## (2번, 5번) = (X, X) → "노트북 화면 각도 조절" 메시지
            elif shoulder_points[5] == None:
                print("양쪽 어깨가 화면에 보이도록 자세를 조정해주세요!")
                
            else:
                pass
            
        ## shoulder_result에 mean_shoulder값 추가
        shoulder.append(mean_shoulder) 

  #------------------------------Face Detection-------------------------------#

        ## 광대 양 끝, 턱 끝
        face_table = [0, 8, 16]
        
        for face in faces:
            landmarks = predictor(image, face)
            
            for n in face_table:
                y = landmarks.part(n).y
                face_points.append(y)

  #----------------------------거북목 자세 판단---------------------------------#

        ## Detection Data 거리 관계
        ### Face
        test_face_ = pd.DataFrame(face_points)
        test_face = test_face_.to_numpy()
        ### Pose : Shoulder
        test_shoulder_ = pd.DataFrame(shoulder)
        test_shoulder = test_shoulder_.to_numpy()
        ### 거리 관계 : 차
        subtract = test_face.T - test_shoulder
        
        ## SVM Classifier
        X_test = pd.DataFrame(subtract)
        y_test = svm_from_joblib.predict(X_test)                           # 학습된 pkl 파일로 변경
    
        # 거북목 자세인가?
        ## 바른 자세(0) → 광대 거리(X_length_low)
        if y_test == 0:
            y_vec.append(int(y_test))
            
            for face in faces:
                landmarks = predictor(image, face)
                
            x1 = landmarks.part(0).x
            x17 = landmarks.part(16).x
            X_length = x17 - x1
            X_length_low.append(int(X_length))
            print("X_length_low : ",X_length_low)
            
        ## 거북목 자세(1) → 거북목 자세 누적(y_vec)
        elif y_test == 1:
            y_vec.append(int(y_test))
            
        else:
            pass
    
        ## y_vec 값 확인
        print("y_vec : ",y_vec)
        
    else:
        pass
    # 종료    
    cap.release()
    cv2.destroyAllWindows()


# #### 2) High 일 때

# In[10]:


def detection_high():
     
  #---------------------------------Setting-----------------------------------#
    
    # Face Detection : 1번, 17번
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # 얼굴이 화면에 보이는가?
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)           # [WARN:0] global cap Error
    time.sleep(1)
 
  #-----------------------------Webcam 실행---------------------------------#
    
    ## Webcam이 실행하지 않는 경우 → 종료
    if not cap.isOpened():
        exit()
    
    ## Webcam이 실행하는 경우 → 판별 알고리즘
    while(y_vec2.count(1) < 10):
        
        # 변수 Setting    
        ## Empty list to store the detected keypoints
        face_points = []
        
        ### OpenCV로 촬영되는 frame
        hasFrame, image = cap.read()
        cv2.imshow('image', image)
        frameCopy = np.copy(image)
        faces = detector(image)

        if not hasFrame:
            break
 
  #---------------------------Face Detection---------------------------#
        
        for face in faces:
            landmarks = predictor(image, face)

        # high일 때 광대 X_length
        x1 = landmarks.part(0).x
        x17 = landmarks.part(16).x
        X_length_high = x17 - x1
        print("X_length_high : ", X_length_high)

#---------------------------거북목 자세 판단----------------------------#

        # 최대값 threshold
        threshold_high = max(X_length_low) + 5
        
        if X_length_high > threshold_high:
            y_test_high = 1
            y_vec2.append(int(y_test_high))
        else:
            y_test_high = 0
            y_vec2.append(int(y_test_high))
            
        ## y_vec 값 확인
        print("X_length_high : ",X_length_high)
        print("y_vec2 : ",y_vec2)    

    else:
        pass
    
    # 종료    
    cap.release()
    cv2.destroyAllWindows()


# ### 4. 실시간 자세 판별 (Low & High)

# In[12]:


"""# Arduino 실행을 위한 통신
arduino = serial.Serial("COM3", 9600)
    
# 거치대 실행을 위한 Detection
detection_low()
mode = 'L'
print("mode : ", mode)

while True:
    # 거북목 자세가 5번 누적
    if mode == 'L':

        ## Arduino 실행
        arduino.write(b'1')
        
        while True:
            c = input()
            
            # High일 때
            detection_high()
            mode = 'H'
            print("mode : ", mode)
            
            if c == 'q':
                break
            else:
                ## Arduino 실행
                arduino.write(b'2')

        else:
            print("종료하겠습니다.")
            
        
        sys.exit()
        
    else:
        pass
else:
    pass"""


# ### 4'. GUI로 실시간 제어

# In[5]:


from tkinter import *
import tkinter as tk

class App:
    def __init__(self):
        
        ## 창 크기 조절
        window = tk.Tk()
        window.geometry("40x40")
        
        ## GUI로 판별 알고리즘 : Start
        startB = Button(window, text = "Start", 
                       command = self.test, 
                       fg="blue", width=10, height=5)
        startB.pack(side=LEFT)
        
        ## 창 종료 : Quit
        quitB = Button(window, text = "Quit", 
                       command = window.destroy, 
                       width=10, height=5)
        quitB.pack(side=RIGHT)
        
        ## 화면에 Pop UP
        window.mainloop()
        
    def test(self):
        # Arduino 실행을 위한 통신
        arduino = serial.Serial("COM3", 9600)

        # 거치대 실행을 위한 Detection
        detection_low()
        mode = 'L'
        print("mode : ", mode)

        while True:
            # 거북목 자세가 5번 누적
            if mode == 'L':

                ## Arduino 실행
                arduino.write(b'1')
                
                while True:
                    c = input()
                    
                    if c != 'q':
                        # High일 때
                        detection_high()
                        mode = 'H'
                        print("mode : ", mode)

                        ## Arduino 실행
                        arduino.write(b'2')
                        
                        ## Global 변수 초기화 [Error 해결]
                        del y_vec2[:]
                        del X_length_high[:]
                    else:
                        break
                        
                else:
                    pass
            else:
                pass
        else:
            pass


# In[ ]:


App()

