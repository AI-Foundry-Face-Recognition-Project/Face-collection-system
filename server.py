import cv2
import os
import time
import numpy as nps
import numpy as np
import torch
import base64
# import cupy as np
from helper import read_pkl_model, start_up_init, encode_image
import face_embedding
import face_detector
import sqlinfo as sql
# In[]:
is_real_time=False
VIDEO_PATH='B1_Cam1_1.mp4'
SAVE_VIDEO=False
PROBA_THRESHOLD=0.5
VIDEO_OUT_PATH='output_2.avi'
VIDEO_RESOLUTION=(1920,1080)
VIDEO_FRAME=15.0
SHOW_VIDEO=True
USE_DEFAULT_FACE_DATA=False # if True, use face_vector.npy, else use real time face data
fourcc = cv2.VideoWriter_fourcc(*'XVID')
NPY_PATH='face_vector.npy'
RETINA_THRESHOLD=.85
# In[]:
import mysql.connector
# import password as pw
maxdb = mysql.connector.connect(
    host = sql.host,
    user = sql.user,
    password = sql.password,
    database = sql.database,
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
def sql_find_last_face_img_id():
    cursor.execute("SELECT face_id,CAST(face_id AS UNSIGNED) as face_id_int FROM face order by face_id_int DESC LIMIT 0 , 1;")
    result = cursor.fetchall()
    return int(result[0][0])
def sql_find_last_origin_img_id():
    cursor.execute("SELECT origin_img_id,CAST(origin_img_id AS UNSIGNED) as origin_img_id_int FROM origin_img order by origin_img_id_int DESC LIMIT 0 , 1;")
    result = cursor.fetchall()
    return int(result[0][0])
def sql_find_last_group_id():
    cursor.execute("SELECT NTR_id,CAST(NTR_id AS UNSIGNED) as NTR_id_int FROM NTR order by NTR_id_int DESC LIMIT 0 , 1;")
    result = cursor.fetchall()
    return int(result[0][0])
def sql_write_origin_img(frame):
    frame_str=image2string(frame)
    time_tmp = time.localtime(time.time())
    last_num=sql_find_last_origin_img_id()
    last_num+=1
    time_now=str(time_tmp.tm_year)+"-"+str(time_tmp.tm_mon)+"-"+str(time_tmp.tm_mday)+" "+str(time_tmp.tm_hour)+":"+str(time_tmp.tm_min)+":"+str(time_tmp.tm_sec)
    cursor.execute("insert into origin_img(origin_img_id,img,img_time) VALUES ('%s','%s','%s');"%(str(last_num),frame_str,time_now))
    maxdb.commit()
def sqlwrite_face_img(img,origin_img_id):
    last_num=str(sql_find_last_face_img_id())
    img_str=image2string(img)
    cursor.execute("insert into face(face_id,origin_img_id,face_img) VALUES ('%s','%s','%s');"%(last_num,origin_img_id,img_str))
    maxdb.commit()
    return last_num
def sql_write_NTR(group_id,face_id): # NTR: need to recognize
    cursor.execute("SELECT NTR_id from NTR WHERE NTR_id = %s;"%(group_id))
    result = cursor.fetchall()
    if result == []:
        cursor.execute("insert into NTR(NTR_id) VALUES ('%s');"%(group_id))
        maxdb.commit()
    sql_write_ntr_face_id(group_id,face_id)
def sql_write_ntr_face_id(group_id,face_id):
    cursor.execute("INSERT INTO ntr_face_id (ntr_id, face_id) VALUES ('%s','%s')"%(group_id,face_id))
    maxdb.commit()
# def sql_write_face(vector,img,group_id,origin_img_id):
#     for i,g_id in enumerate(group_id):
#         o_id=origin_img_id[i]
#         face_id=sqlwrite_face_img(img[i],origin_img_id)
#         sql_write_NTR(group_id[i],face_id)
def caculate_distmat(qf, gf):
    m, n = qf.shape[0], gf.shape[0]
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    return distmat
def find_topK(embedding):
    global vector
    qf=torch.tensor([embedding])
    vector_=[]
    if isinstance(vector, torch.Tensor):
        vector_=vector
    else:
        vector_=torch.tensor([vector])
    if vector_.shape[0]==0:
        return "first picture"
    else:
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
def detect_frame(frame):
    global vector#每個人臉的特徵向量
    global face_img#每個臉的原始圖片
    global id#人的編號 [1,2,4,1]
    global origin_img_id#這張人臉是從哪個人臉出來的(origin_img)
    global ntr_id
    imgs=retina(frame)
    embedding_img=[]
    sql_write_origin_img(frame)
    if not(imgs is None):
        for imgs_ in imgs:
            img,box=imgs_['img'],imgs_['box']
            embedding=get_embedding(img)
            re = find_topK(embedding)
            topid,_,proba='','',''
            if re=="first picture":
                ntr_id=sql_find_last_group_id()
                proba=-1
                pass
            else:
                topid,_,proba=re
            if isinstance(vector, torch.Tensor):
                if not isinstance(embedding, torch.Tensor):
                    embedding=torch.tensor([embedding])
                vector=torch.cat((vector,embedding),0)
            elif isinstance(vector, list):
                if not isinstance(embedding, list):
                    embedding=[embedding]
                vector.append(embedding)
            elif isinstance(vector, np.ndarray):
                if not isinstance(embedding, np.ndarray):
                    embedding=np.array([embedding])
                vector = np.concatenate((vector, embedding), axis=0)
            face_img.append(img)
            id.append(topid)
            origin_img_id.append(int(sql_find_last_origin_img_id()))
            if proba > PROBA_THRESHOLD:
                id.append(topid)
            else:
                ntr_id+=1
                id.append(ntr_id)
            #embedding_={'img':img,'box':box,'topid':topid}#,'distance':distance}
            embedding_={'box':box,'topid':topid}
            #print(embedding_)
            embedding_img.append(embedding_)
            
    else:
        print("@@")
        for x,y,z,w in zip(vector,face_img,id,origin_img_id):
            print("z")
            sql_write_NTR(z,sqlwrite_face_img(y,w))
        vector=[]
        face_img=[]
        id=[]
        origin_img_id=[]
    return embedding_img
if __name__ == "__main__":
    cap=cv2.VideoCapture(VIDEO_PATH)
    VIDEO_FRAME=cap.get(cv2.CAP_PROP_FRAME_COUNT) if is_real_time else None
    out = cv2.VideoWriter(VIDEO_OUT_PATH, fourcc, VIDEO_FRAME, VIDEO_RESOLUTION) if SAVE_VIDEO else None
    while True:
        ret, frame = cap.read()
        frame = frame[0:1080, 200:1500]
        start_time=time.time()
        if not ret:
            break
        dectcts_re=detect_frame(frame)
        for dectcts in dectcts_re:
            box,topid=dectcts['box'],dectcts['topid']
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        cv2.imshow('frame', frame)if SHOW_VIDEO else None
        out.write(frame) if SAVE_VIDEO else None
        if is_real_time:
            #for i in range(int((time.time()-start_time)*VIDEO_FRAME)):
            #    cap.read()
            pass
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release() if SAVE_VIDEO else None
    cv2.destroyAllWindows()