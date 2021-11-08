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
  "return_face_data": True,
  "return_landmarks": True,
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
print("face numbers:")
print(len(images[0]['faces']))
print("first face imformations:")
print(images[0]['faces'][0])
print("first face bbox:")
print(images[0]['faces'][0]['bbox'])#left-up x,y, right-down x,y
print("first face 512dims feature:")
print(images[0]['faces'][0]['vec'])#left-up x,y, right-down x,y
