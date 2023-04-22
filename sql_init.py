import mysql.connector
import cv2
import base64
import numpy as np
import sqlinfo as sql
maxdb = mysql.connector.connect(
    host = sql.host,
    user = sql.user,
    password = sql.password,
    database = sql.database,
    )


def D_BASE64(Str):
    if(len(Str)%4==1):
        Str+="==="
    elif(len(Str)%4==2):
        Str+="=="
    elif(len(Str)%4==3):
        Str+="="
    return Str
def string2image(img_str):
    img_str=D_BASE64(img_str)
    print(img_str)
    img_str = base64.b64decode(img_str)
    img_arr = np.frombuffer(img_str, np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    return img
def image2string(image):
    img_str = cv2.imencode('.jpg', image)[1].tostring()
    img_str = base64.b64encode(img_str).decode('utf-8')
    return img_str
def sql_write_ntr():
    cursor.execute("INSERT INTO ntr (ntr_id) VALUES ('0')")
    maxdb.commit()
    sql_write_ntr_face_id()
def sql_write_ntr_face_id():
    cursor.execute("INSERT INTO ntr_face_id (ntr_id, face_id) VALUES ('0','0')")
    maxdb.commit()
image="unknown.jpg"
image1= cv2.imread(image)
image_str=image2string(image1)
cursor=maxdb.cursor()
cursor.execute("insert into origin_img(origin_img_id,img,img_time) VALUES ('0','%s','0001-01-01 00:00:00');" %image_str)
cursor.execute("insert into face(face_id,origin_img_id,face_img) VALUES ('0','0','%s');" %image_str)
maxdb.commit()
sql_write_ntr()



