import mysql.connector
import cv2
import base64
import numpy as nps
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
    img_arr = nps.frombuffer(img_str, nps.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    return img
image_1=string2image(image1)
image_2=string2image(image2)
image_3=string2image(image3)
maxdb = mysql.connector.connect(
    host = "127.0.0.1",
    user = "root",
    password = "!A1078d2906e2a",
    database = "proj",
    )
cursor=maxdb.cursor()
cursor.execute("insert into origin_img(origin_img_id,img,img_time) VALUES ('1','1','0001-01-01 00:00:00');")
cursor.execute("insert into origin_img(origin_img_id,img,img_time) VALUES ('2','1','0001-01-01 00:00:00');")
cursor.execute("insert into origin_img(origin_img_id,img,img_time) VALUES ('3','1','0001-01-01 00:00:00');")
cursor.execute("insert into face(face_id,origin_img_id,face_img) VALUES ('1','1','%s');" % image1)
cursor.execute("insert into face(face_id,origin_img_id,face_img) VALUES ('2','2','%s');" % image2)
cursor.execute("insert into face(face_id,origin_img_id,face_img) VALUES ('3','3','%s');" % image3)
