import requests
import json 
import time
import base64
import ujson

def file2base64(path):
    with open(path, mode='rb') as fl:
        encoded = base64.b64encode(fl.read()).decode('ascii')
        return encoded

# embedding demo
url = "http://localhost:18081/embedding"

files = ['src/api_trt/test_images/TH.png']


target = file2base64(files[0])

params={
  "images": {
    "data": [
      target
    ]
  },
  "threshold": 0.6,
  "embed_only": False,
  "return_face_data": False,
  "return_landmarks": False,
  "extract_embedding": True,
  "extract_ga": True,
  "limit_faces": 0,
  "min_face_size": 0,
  "verbose_timings": True,
  "api_ver": "2"
}


result = requests.post(url,json.dumps(params))
content = ujson.loads(result.text)

took = content.get('took')
status = content.get('status')
images = content.get('data')
#print(images[0]['vec'])

# fd demo
url = "http://localhost:18081/fd"

files = ['src/api_trt/test_images/lumia.jpg']


target = file2base64(files[0])

params={
  "images": {
    "data": [
      target
    ]
  },
  "threshold": 0.6,
  "embed_only": False,
  "return_face_data": True,
  "return_landmarks": True,
  "extract_embedding": False,
  "extract_ga": True,
  "limit_faces": 0,
  "min_face_size": 0,
  "verbose_timings": True,
  "api_ver": "2"
}


result = requests.post(url,json.dumps(params))
content = ujson.loads(result.text)

took = content.get('took')
status = content.get('status')
images = content.get('data')
print("face numbers:")
#print(len(images[0]['faces']))
print("first face imformations:")
#print(images[0]['faces'][0])
print("first face bbox:")
print(images[0]['faces'][0]['bbox'])#left-up x,y, right-down x,y

# fd+fr demo
url = "http://localhost:18081/extract"

files = ['src/api_trt/test_images/Stallone.jpg']


target = file2base64(files[0])

params={
  "images": {
    "data": [
      target
    ]
  },
  "threshold": 0.6,
  "embed_only": False,
  "return_face_data": False,
  "return_landmarks": False,
  "extract_embedding": True,
  "extract_ga": False,
  "limit_faces": 0,
  "min_face_size": 0,
  "verbose_timings": True,
  "api_ver": "2",
  "msgpack": False
}


result = requests.post(url,json.dumps(params))
content = ujson.loads(result.text)

took = content.get('took')
status = content.get('status')
images = content.get('data')
print("face numbers:")
print(len(images[0]['faces']))
print("first face imformations:")
print(images[0]['faces'][0])
print("first face bbox:")
print(images[0]['faces'][0]['bbox'])#left-up x,y, right-down x,y
print("first face 512dims feature:")
print(images[0]['faces'][0]['vec'])#left-up x,y, right-down x,y

# webcam demo
import cv2
from numpy import dot
from numpy.linalg import norm

import listname as SSS

# 選擇第二隻攝影機
cap = cv2.VideoCapture(0)
import numpy as np
while(True):
  # 從攝影機擷取一張影像
  ret, frame = cap.read()

  jpg_as_text = base64.b64encode(cv2.imencode('.jpg', frame)[1]).decode()
  params={
   "images": {
     "data": [
       jpg_as_text
     ]
   },
   "threshold": 0.6,
   "embed_only": False,
   "return_face_data": False,
   "return_landmarks": False,
   "extract_embedding": True,
   "extract_ga": False,
   "limit_faces": 0,
   "min_face_size": 0,
   "verbose_timings": True,
   "api_ver": "2",
   "msgpack": False
  }
  result = requests.post(url,json.dumps(params))
  content = ujson.loads(result.text)
  
  took = content.get('took')
  status = content.get('status')
  images = content.get('data')
  # face number
  #print(len(images[0]['faces']))
  for i in range(len(images[0]['faces'])):

      # draw face bbox
      cv2.rectangle(frame, (images[0]['faces'][i]['bbox'][0], images[0]['faces'][i]['bbox'][1]), (images[0]['faces'][i]['bbox'][2], images[0]['faces'][i]['bbox'][3]), (0, 0, 255), 2, cv2.LINE_AA)  
      # write the 512 feature in picture
      cv2.putText(frame, str(images[0]['faces'][i]['vec']), (images[0]['faces'][i]['bbox'][0], images[0]['faces'][i]['bbox'][1]), cv2.FONT_HERSHEY_SIMPLEX,  1, (0, 255, 0), 1, cv2.LINE_AA)
      # example for match face
      #cos_sim = dot(feature1, images[0]['faces'][i]['vec'])/(norm(feature1)*norm(images[0]['faces'][i]['vec']))
      #if cos_sim > 0.6:
      #    print("it the same")
      

              
              
  cv2.imshow('frame', frame)
  
  # 若按下 q 鍵則離開迴圈
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# 釋放攝影機
cap.release()

# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()
