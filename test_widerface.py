import os
import base64
import requests
import glob
import time
import multiprocessing
import numpy as np
from itertools import chain, islice
import ujson
import logging
import shutil
import cv2
from tqdm import tqdm
import time
# cos_sim
from sklearn.metrics.pairwise import cosine_similarity

dir_path = os.path.dirname(os.path.realpath(__file__))
test_cat = os.path.join(dir_path, 'images')

session = requests.Session()
session.trust_env = False

logging.basicConfig(
    level='INFO',
    format='%(asctime)s %(levelname)s - %(message)s',
    datefmt='[%H:%M:%S]',
)

def file2base64(path):
    with open(path, mode='rb') as fl:
        encoded = base64.b64encode(fl.read()).decode('ascii')
        return encoded

def save_crop(data, name):
    img = base64.b64decode(data)
    with open(name, mode="wb") as fl:
        fl.write(img)
        fl.close()

def extract_vecs(task):
    target = task[0]
    server = task[1]
    images = dict(data=target)
    req = dict(images=images,
               threshold=0.5,
               extract_ga=False,
               extract_embedding=False,
               return_face_data=True,
               embed_only=False, # If set to true API expects each image to be 112x112 face crop
               limit_faces=0, # Limit maximum number of processed faces, 0 = no limit
               api_ver='2'
               )

    resp = session.post(server, json=req, timeout=120)

    content = ujson.loads(resp.content)
    took = content.get('took')
    status = content.get('status')
    images = content.get('data')
    counts = [len(e.get('faces', [])) for e in images]
    a_recode=[]  
    for im in images:
        faces = im.get('faces', [])
        return faces

def to_chunks(iterable, size=10):
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator, size - 1))


if __name__ == "__main__":

    ims = 'src/api_trt/test_images'
    server = 'http://localhost:18081/extract'

    if os.path.exists('crops'):
        shutil.rmtree('crops')
    os.mkdir('crops')

    speeds = []
    
    # Test cam
    cap = cv2.VideoCapture("1111.mp4")
    yourPath = './WIDER_val/images/'
    allFileList = os.listdir(yourPath)

    for filex in allFileList: #search all val subset
        if not os.path.isdir('./result/'+filex):
            os.mkdir('./result/'+filex)    
        
        inner_allFileList = os.listdir(yourPath+filex)
        for filey in inner_allFileList: #search all jpg subset
            
            f = open('./result/'+filex+'/'+filey.split('.')[0]+".txt", "a")
            f.write(filey.split('.')[0]+'\n')
            print(filey.split('.')[0])
            image = cv2.imread(yourPath+filex+'/'+filey,1)
            #cv2.imshow("1",image)
            print("image.shape[0] ",image.shape[0] )#h
            w = image.shape[0] / image.shape[0]
            h = image.shape[1] / image.shape[1] 

            #image=cv2.resize(image,(640,480))
            target = [base64.b64encode(cv2.imencode('.jpg', image)[1]).decode()]
            print("done pre-prosscess")
        
            target_chunks = to_chunks(target, 1)
            task_set = [[list(chunk), server] for i, chunk in enumerate(target_chunks)]
            task_set = list(task_set)
            print('Encoding images.... Finished')
            pool = multiprocessing.Pool(2)

            t0 = time.time()
            r = pool.map(extract_vecs, task_set)
            f.write(str(len(r[0]))+'\n')
            if r != None:
                for i, face in enumerate(r[0]):
                    bbox = face.get('bbox')
                    prob = face.get('prob')
                    
                    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    f.write(str(float(bbox[0]*h))+' '+ str(float(bbox[1]*w))+' '+ str(float(bbox[2]-bbox[0])*h)+' '+ str(float(bbox[3]-bbox[1])*w)+' '+ str(prob)+'\n')
                    
            
                    
            f.close()
            t1 = time.time()
            took = t1 - t0
            speed = 1 / took
            speeds.append(speed)
            print("Took: {} ({} im/sec)".format(took, speed))

            pool.close()
            mean = np.mean(speeds)
            median = np.median(speeds)

            print(f'mean: {mean} im/sec\n'
                f'median: {median}\n'
                f'min: {np.min(speeds)}\n'
                f'max: {np.max(speeds)}\n'
                )

            cv2.imshow("1",image)  
            #time.sleep(50)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break              
    cap.release()
