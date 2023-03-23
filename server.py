import cv2
import os
import time
import numpy as nps
import torch
import base64
# import cupy as np
from helper import read_pkl_model, start_up_init, encode_image
import face_embedding
import face_detector
# In[]:
is_real_time=False
VIDEO_PATH='b1_cam1_2.avi'
SAVE_VIDEO=True
PROBA_THRESHOLD=0.5
VIDEO_OUT_PATH='output_2.avi'
VIDEO_RESOLUTION=(1920,1080)
VIDEO_FRAME=15.0
SHOW_VIDEO=True
USE_DEFAULT_FACE_DATA=True # if True, use face_vector.npy, else use real time face data
fourcc = cv2.VideoWriter_fourcc(*'XVID')
NPY_PATH='face_vector.npy'

# In[]:
import mysql.connector
# import password as pw
maxdb = mysql.connector.connect(
    host = "127.0.0.1",
    user = "root",
    password = "Iloveyou",
    database = "topic",
    )
cursor=maxdb.cursor()

args=start_up_init()
npy=nps.load(NPY_PATH,allow_pickle=True)
id=[]
vector=[]
face_img=[]
origin_img_id=[]
if USE_DEFAULT_FACE_DATA:
    for i in npy:
        vector.append(i[2])
        id.append(i[0])
vector=torch.tensor(vector)
detector = face_detector.DetectorModel(args)
embedding = face_embedding.EmbeddingModel(args)
get_embedding=lambda img: embedding.get_one_feature(img)
def image2string(image):
    img_str = cv2.imencode('.jpg', image)[1].tostring()
    img_str = base64.b64encode(img_str).decode('utf-8')
    return img_str
def string2image(img_str):
    img_str = base64.b64decode(img_str)
    img_arr = nps.frombuffer(img_str, nps.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    return img
def sql_find_last_img_id():
    cursor.execute("SELECT face_id FROM face order by 'face_id' DESC LIMIT 0 , 1;")
    result = cursor.fetchall()
<<<<<<< HEAD
    return result

def sql_find_last_img_id():
    cursor.execute("SELECT origin_img_id FROM origin_img order by 'origin_img_id' DESC LIMIT 0 , 1;")
    result = cursor.fetchall()
    return result

def sql_find_last_group_id():
    cursor.execute("SELECT NTR_id FROM NTR order by 'NTR_id' DESC LIMIT 0 , 1;")
    result = cursor.fetchall()
    return result

=======
    return result#check
def sql_find_last_img_id():
    cursor.execute("SELECT origin_img_id FROM origin_img order by 'origin_img_id' DESC LIMIT 0 , 1;")
    result = cursor.fetchall()
    return result#check
def sql_find_last_group_id():
    cursor.execute("SELECT NTR_id FROM NTR order by 'NTR_id' DESC LIMIT 0 , 1;")
    result = cursor.fetchall()
    return result#check
>>>>>>> 1dc6d48ef4c9f0c1ea145661953d367856c5643b
def sql_write_origin_img(frame):
    frame_str=image2string(frame)
    time_tmp = time.localtime(time.time())
    last_num = str(int(sql_find_last_img_id())+1)
    time_now=str(time_tmp.tm_year)+"-"+str(time_tmp.tm_mon)+"-"+str(time_tmp.tm_mday)+" "+str(time_tmp.tm_hour)+":"+str(time_tmp.tm_min)+":"+str(time_tmp.tm_sec)
    cursor.execute("insert into origin_img(origin_img_id,img,img_time) VALUES ('%s','%s','%s');"%(last_num,frame_str,time_now))
<<<<<<< HEAD

=======
    #check
>>>>>>> 1dc6d48ef4c9f0c1ea145661953d367856c5643b
def sqlwrite_face_img(img,origin_img_id):
    last_num=sql_find_last_img_id()
    img_str=image2string(img)
    cursor.execute("insert into face(face_id,origin_img_id,face_img) VALUES ('%s','%s','%s');"%(last_num,origin_img_id,img_str))
    return last_num
def sql_write_NTR(group_id,face_id): # NTR: need to recognize
    cursor.execute("insert into NTR(NTR_id,face_id) VALUES ('%s','%s');"%(group_id,face_id))
def sql_write_face(vector,img,group_id,origin_img_id):
    for i,g_id in enumerate(group_id):
        o_id=origin_img_id[i]
        face_id=sqlwrite_face_img(img[i],origin_img_id)
        sql_write_NTR(group_id[i],face_id)
def caculate_distmat(qf, gf):
    m, n = qf.shape[0], gf.shape[0]
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    return distmat
def find_topK(embedding):
    global vector
    qf=torch.tensor([embedding])
    vector_=torch.tensor([vector])
    distmat=caculate_distmat(qf,vector_)
    a,b=torch.topk(distmat,1,largest=False)
    top=[id[b],b,a]
    return top
def retina(frame):
    threshold=args.embedding_threshold
    imgs=[]
    for img, box in detector.get_all_boxes(frame, save_img=False):
        if box[4] > threshold:
            imgs.append({'img':img,'box':box})
    return imgs
#     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                       _oo0oo_
#                      o8888888o
#                      88" . "88
#                      (| -_- |)
#                      0\  =  /0
#                    ___/`---'\___
#                  .' \\|     |// '.
#                 / \\|||  :  |||// \
#                / _||||| -:- |||||- \
#               |   | \\\  -  /// |   |
#               | \_|  ''\---/''  |_/ |
#               \  .-\__  '-'  ___/-. /
#             ___'. .'  /--.--\  `. .'___
#          ."" '<  `.___\_<|>_/___.' >' "".
#         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#         \  \ `_.   \_ __\ /__ _/   .-` /  /
#     =====`-.____`.___ \_____/___.-`___.-'=====
#                       `=---='
#
#
#     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def detect_frame(frame):
    global vector
    global face_img
    global id
    global origin_img_id
    imgs=(frame)
    o_id=sql_find_last_img_id()
    embedding_img=[]
    if imgs:
        
        for imgs_ in imgs:
            img,box=imgs_['img'],imgs_['box']
            embedding=get_embedding(img)
            topid,_,proba = find_topK(embedding)
            vector.append(embedding)
            face_img.append(img)
            id.append(topid)
            origin_img_id.append(o_id)
            if proba > PROBA_THRESHOLD:
                id.append(topid)
            else:
                id.append(sql_find_last_group_id()+1)
            #embedding_={'img':img,'box':box,'topid':topid}#,'distance':distance}
            embedding_={'box':box,'topid':topid}
            print(embedding_)
            embedding_img.append(embedding_)
    else:
        sql_write_face(vector,img,id,origin_img_id)
        vector=[]
    return embedding_img

if __name__ == "__main__":
    cap=cv2.VideoCapture(VIDEO_PATH)
    VIDEO_FRAME=cap.get(cv2.CAP_PROP_FRAME_COUNT) if is_real_time else None
    out = cv2.VideoWriter(VIDEO_OUT_PATH, fourcc, VIDEO_FRAME, VIDEO_RESOLUTION) if SAVE_VIDEO else None
    while True:
        ret, frame = cap.read()
        start_time=time.time()
        if not ret:
            break
        dectcts_re=detect_frame(frame)
        for dectcts in dectcts_re:
            box,topid=dectcts['box'],dectcts['topid']
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        if dectcts_re:
            sql_write_origin_img(frame)
        cv2.imshow('frame', frame)if SHOW_VIDEO else None
        out.write(frame) if SAVE_VIDEO else None
        for i in range(int((time.time()-start_time)*VIDEO_FRAME)):
            cap.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release() if SAVE_VIDEO else None
    cv2.destroyAllWindows()