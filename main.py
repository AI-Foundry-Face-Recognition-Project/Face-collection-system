# -*- coding: utf-8 -*-retranslateUi

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!
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
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QApplication, QMessageBox)
from PyQt5.QtGui import *
from PyQt5.QtCore import QTimer
import mysql.connector
import cv2
import base64
import numpy as np
import sqlinfo as sql
import time
maxdb = mysql.connector.connect(
    host = sql.host,
    user = sql.user,
    password = sql.password,
    database = sql.database,
    )
cursor=maxdb.cursor()

class Ui_MainWindow(object):
    def __init__(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.user_idle)
        self.RELOAD_TIME=60*1000 #update per minute
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(480, 320)
        MainWindow.setMinimumSize(QtCore.QSize(480, 320))
        MainWindow.setMaximumSize(QtCore.QSize(480, 320))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setMinimumSize(QtCore.QSize(480, 320))
        self.centralwidget.setMaximumSize(QtCore.QSize(480, 16777215))
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 10, 481, 321))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap(":/background/background.png"))
        self.label.setObjectName("label")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(240, 100, 201, 40))
        self.ImgDisp = QtWidgets.QLabel(self.centralwidget)
        self.ImgDisp.setGeometry(QtCore.QRect(60, 50, 100, 110))
        self.ImgDisp.setObjectName("ImgDisp")
        self.ImgDisp.show()
        self.ImgDisp.raise_()
        font = QtGui.QFont()
        font.setPointSize(25)
        font.setBold(True)
        font.setItalic(False)
        self.lineEdit.setFont(font)
        self.lineEdit.setStyleSheet("color: rgb(255, 221, 95);\n"
"background-color: rgb(21, 35, 70);\n"
"")
        self.lineEdit.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit.setObjectName("lineEdit")
        self.Button_1 = QtWidgets.QPushButton(self.centralwidget)
        self.Button_1.setGeometry(QtCore.QRect(0, 220, 60, 50))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        self.Button_1.setFont(font)
        self.Button_1.setStyleSheet("color: rgb(255, 221, 95);\n"
"background-color: rgb(55, 176, 219);")
        self.Button_1.setObjectName("Button_1")
        self.Button_2 = QtWidgets.QPushButton(self.centralwidget)
        self.Button_2.setGeometry(QtCore.QRect(60, 220, 60, 50))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        self.Button_2.setFont(font)
        self.Button_2.setStyleSheet("color: rgb(255, 221, 95);\n"
"background-color: rgb(55, 176, 219);")
        self.Button_2.setObjectName("Button_2")
        self.Button_3 = QtWidgets.QPushButton(self.centralwidget)
        self.Button_3.setGeometry(QtCore.QRect(120, 220, 60, 50))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        self.Button_3.setFont(font)
        self.Button_3.setStyleSheet("color: rgb(255, 221, 95);\n"
"background-color: rgb(55, 176, 219);")
        self.Button_3.setObjectName("Button_3")
        self.Button_4 = QtWidgets.QPushButton(self.centralwidget)
        self.Button_4.setGeometry(QtCore.QRect(180, 220, 60, 50))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        self.Button_4.setFont(font)
        self.Button_4.setStyleSheet("color: rgb(255, 221, 95);\n"
"background-color: rgb(55, 176, 219);")
        self.Button_4.setObjectName("Button_4")
        self.Button_5 = QtWidgets.QPushButton(self.centralwidget)
        self.Button_5.setGeometry(QtCore.QRect(240, 220, 60, 50))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        self.Button_5.setFont(font)
        self.Button_5.setStyleSheet("color: rgb(255, 221, 95);\n"
"background-color: rgb(55, 176, 219);")
        self.Button_5.setObjectName("Button_5")
        self.Button_6 = QtWidgets.QPushButton(self.centralwidget)
        self.Button_6.setGeometry(QtCore.QRect(0, 270, 60, 50))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        self.Button_6.setFont(font)
        self.Button_6.setStyleSheet("color: rgb(255, 221, 95);\n"
"background-color: rgb(55, 176, 219);")
        self.Button_6.setObjectName("Button_6")
        self.Button_7 = QtWidgets.QPushButton(self.centralwidget)
        self.Button_7.setGeometry(QtCore.QRect(60, 270, 60, 50))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        self.Button_7.setFont(font)
        self.Button_7.setStyleSheet("color: rgb(255, 221, 95);\n"
"background-color: rgb(55, 176, 219);")
        self.Button_7.setObjectName("Button_7")
        self.Button_8 = QtWidgets.QPushButton(self.centralwidget)
        self.Button_8.setGeometry(QtCore.QRect(120, 270, 60, 50))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        self.Button_8.setFont(font)
        self.Button_8.setStyleSheet("color: rgb(255, 221, 95);\n"
"background-color: rgb(55, 176, 219);")
        self.Button_8.setObjectName("Button_8")
        self.Button_9 = QtWidgets.QPushButton(self.centralwidget)
        self.Button_9.setGeometry(QtCore.QRect(180, 270, 60, 50))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        self.Button_9.setFont(font)
        self.Button_9.setStyleSheet("color: rgb(255, 221, 95);\n"
"background-color: rgb(55, 176, 219);")
        self.Button_9.setObjectName("Button_9")
        self.Button_0 = QtWidgets.QPushButton(self.centralwidget)
        self.Button_0.setGeometry(QtCore.QRect(240, 270, 60, 50))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        self.Button_0.setFont(font)
        self.Button_0.setStyleSheet("color: rgb(255, 221, 95);\n"
"background-color: rgb(55, 176, 219);")
        self.Button_0.setObjectName("Button_0")
        self.Button_del = QtWidgets.QPushButton(self.centralwidget)
        self.Button_del.setGeometry(QtCore.QRect(300, 270, 121, 50))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        self.Button_del.setFont(font)
        self.Button_del.setStyleSheet("color: rgb(255, 221, 95);\n"
"background-color: rgb(55, 176, 219);")
        self.Button_del.setObjectName("Button_del")
        self.Button_enter = QtWidgets.QPushButton(self.centralwidget)
        self.Button_enter.setGeometry(QtCore.QRect(421, 270, 60, 50))
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        self.Button_enter.setFont(font)
        self.Button_enter.setStyleSheet("color: rgb(255, 221, 95);\n"
"background-color: rgb(55, 176, 219);")
        self.Button_enter.setObjectName("Button_enter")
        self.Button_next = QtWidgets.QPushButton(self.centralwidget)
        self.Button_next.setGeometry(QtCore.QRect(360, 220, 61, 50))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        self.Button_next.setFont(font)
        self.Button_next.setStyleSheet("color: rgb(255, 221, 95);\n"
"background-color: rgb(55, 176, 219);")
        self.Button_next.setObjectName("Button_next")
        self.Button_priview = QtWidgets.QPushButton(self.centralwidget)
        self.Button_priview.setGeometry(QtCore.QRect(300, 220, 60, 50))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        self.Button_priview.setFont(font)
        self.Button_priview.setStyleSheet("color: rgb(255, 221, 95);\n"
"background-color: rgb(55, 176, 219);")
        self.Button_priview.setObjectName("Button_priview")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(0, 0, 481, 321))
        self.label_2.setText("")
        self.label_2.setPixmap(QtGui.QPixmap(":/background/backcoler.png"))
        self.label_2.setObjectName("label_2")
        self.Button_reload = QtWidgets.QPushButton(self.centralwidget)
        self.Button_reload.setGeometry(QtCore.QRect(421, 220, 60, 50))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        self.Button_reload.setFont(font)
        self.Button_reload.setStyleSheet("color: rgb(255, 221, 95);\n"
"background-color: rgb(55, 176, 219);")
        self.Button_reload.setObjectName("Button_reload")
        self.label_2.raise_()
        self.label.raise_()
        self.lineEdit.raise_()
        self.Button_1.raise_()
        self.Button_2.raise_()
        self.Button_3.raise_()
        self.Button_4.raise_()
        self.Button_5.raise_()
        self.Button_6.raise_()
        self.Button_7.raise_()
        self.Button_8.raise_()
        self.Button_9.raise_()
        self.Button_0.raise_()
        self.Button_del.raise_()
        self.Button_enter.raise_()
        self.Button_next.raise_()
        self.Button_priview.raise_()
        self.Button_reload.raise_()
        self.ImgDisp.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.image=[]
        self.pic_label=0
        self.user_idle()
    def mousePressEvent(self,event):
        print('mouse press')
        self.timer.stop()
        self.timer.start(self.RELOAD_TIME)
    def user_idle(self):
        print('user idle reloading data')
        self.timer.stop()
        self.timer.start(self.RELOAD_TIME)
        self.reloadData()
    def setupImg(self):
        self.pic_label=0
        self.pic_maxlabel=0
        self.reloadData()
    def sql_sync(self):
        pass
    def image2string(self,image):
        img_str = cv2.imencode('.jpg', image)[1].tostring()
        img_str = base64.b64encode(img_str).decode('utf-8')
        return img_str
    def string2image(self,img_str):
        img_str = base64.b64decode(img_str)
        img_arr = np.frombuffer(img_str, np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        return img
    def sql_find_last_img_id():
        cursor.execute("SELECT origin_img_id,CAST(origin_img_id AS UNSIGNED) as origin_img_id_int FROM origin_img order by origin_img_id_int DESC LIMIT 0 , 1;")
        result = cursor.fetchall()
        return result#check
    def sql_find_face_img(self):
        # cursor.execute("SELECT face_img \
        #                FROM face,ntr,ntr_face_id \
        #                where ntr.ntr_id=ntr_face_id.ntr_id and ntr_face_id.face_id=face.face_id \
        #                order by ntr.ntr_id DESC;")
    
        cursor.execute("SELECT f.face_img,f.face_id,n.NTR_id FROM face as f,NTR_face_id as n where f.face_id in (select face_id from (select Distinct face_id,NTR_id from NTR_face_id) as f) and f.face_id=n.face_id order by n.NTR_id ASC;")
        temp=cursor.fetchall()
        img=[self.string2image(element) for element,_,_ in temp]
        num=[num for _,num,_ in temp]
        result_ntr_id=[ids for _,_,ids in temp]
        result=[]
        result_num=[]
        for id in sorted(list(set(result_ntr_id))):
            tmp=[[img[i],num[i]] for i,j in enumerate(result_ntr_id) if j==id]
            a=tmp[len(tmp)//2]
            result.append(a[0])
            result_num.append(a[1])
        self.pic_maxlabel=len(result)-1
        return result,result_num
    def sql_remove_NTR(self,ntr_id):
        cursor.execute("DELETE FROM NTR_face_id WHERE NTR_id = "+ntr_id+";")
        maxdb.commit()
        cursor.execute("DELETE FROM NTR WHERE NTR_id = "+ntr_id+";")
        maxdb.commit()
    def sql_find_ntr_id (self,face_id):
        cursor.execute("SELECT NTR_id FROM NTR_face_id WHERE face_id = "+face_id+";")
        re=cursor.fetchall()
        return re [0][0]
    def sql_add_people(self,id,ntr_id):
        exist=self.sql_find_people_id(id)
        if not exist:
            cursor.execute("INSERT INTO people (people_id) VALUES ('%s')"%(id))
            maxdb.commit()
        self.sql_write_people_face_id(id,ntr_id)
    def sql_write_people_face_id(self,id,ntr_id):
        result=self.sql_find_allface_by_ntrid(ntr_id)
        for face_id in result:
            cursor.execute("INSERT INTO people_face_id (people_id, face_id) VALUES ('%s','%s');"%(id, face_id[0]))
            maxdb.commit()
    def sql_find_people_id(self,id):
        cursor.execute("SELECT people_id FROM people;")
        results = cursor.fetchall()
        for result  in results:
            if id in result[0]:
                return True
        return False
    def sql_find_allface_by_ntrid(self,ntr_id):
        cursor.execute("SELECT face_id FROM NTR_face_id where NTR_id='"+ntr_id+"';")
        return cursor.fetchall()
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.lineEdit.setText(_translate("MainWindow", "????"))
        self.Button_1.setText(_translate("MainWindow", "1"))
        self.Button_2.setText(_translate("MainWindow", "2"))
        self.Button_3.setText(_translate("MainWindow", "3"))
        self.Button_4.setText(_translate("MainWindow", "4"))
        self.Button_5.setText(_translate("MainWindow", "5"))
        self.Button_6.setText(_translate("MainWindow", "6"))
        self.Button_7.setText(_translate("MainWindow", "7"))
        self.Button_8.setText(_translate("MainWindow", "8"))
        self.Button_9.setText(_translate("MainWindow", "9"))
        self.Button_0.setText(_translate("MainWindow", "0"))
        self.Button_del.setText(_translate("MainWindow", "⌫"))
        self.Button_enter.setText(_translate("MainWindow", "OK"))
        self.Button_next.setText(_translate("MainWindow", "  》"))
        self.Button_priview.setText(_translate("MainWindow", "《  "))
        self.Button_reload.setText(_translate("MainWindow", "⟳"))
        self.ImgDisp.setText(_translate("MainWindow", "."))
    def setLineText(self, text):
        if self.lineEdit.text() == "????":
            self.lineEdit.setText(text)
        else:
            self.lineEdit.setText(self.lineEdit.text() + text)
    def cvimg_to_qtimg(self,cvimg):
        height, width, depth = cvimg.shape
        cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
        cvimg = QImage(cvimg.data, width, height, width * depth, QImage.Format_RGB888)
        return cvimg
    def qtImshow(self,img):
        x,y,z=img.shape
        x_new,y_new=100,110
        if x / y >= x_new / y_new:
            img_new = cv2.resize(img, (x_new, int(y * x_new / x)))
        else:
            img_new = cv2.resize(img, (int(x * y_new / y), y_new))
        QtImg = self.cvimg_to_qtimg(img_new)
        self.ImgDisp.setPixmap(QtGui.QPixmap.fromImage(QtImg))
        size = QtImg.size()
        self.ImgDisp.resize(size)
    def reloadData(self):
        print("start reload")
        time_tmp = time.localtime(time.time())
        time_now=str(time_tmp.tm_year)+"-"+str(time_tmp.tm_mon)+"-"+str(time_tmp.tm_mday)+" "+str(time_tmp.tm_hour)+":"+str(time_tmp.tm_min)+":"+str(time_tmp.tm_sec)
        cursor.execute("insert into reload(img_time) VALUES ('%s');"%(time_now))
        maxdb.commit()
        msgBox = QMessageBox()
        msgBox.setWindowTitle('connecting to server')
        msgBox.setText('更新資料中')
        msgBox.setStandardButtons(QMessageBox.Ok)
        msgBox.button(QMessageBox.Ok).hide()
        msgBox.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        msgBox.button(QMessageBox.Ok).animateClick(1000)
        msgBox.exec_()  
        self.lineEdit.setText("????")
        self.image=[]
        self.image_num=0
        self.image,self.image_num=self.sql_find_face_img()
        self.image.reverse()
        self.image_num.reverse()
        self.pic_label=0
        self.qtImshow(self.image[self.pic_label])
        print("end reload")
    def enter(self):
        check = QMessageBox()
        check.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        reply =check.information(None, '上傳確認', '您好%s\n確定上傳嗎？' % self.lineEdit.text(), QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            if self.sql_find_ntr_id(self.image_num[self.pic_label])=='0':
                msgBox = QMessageBox()
                msgBox.setWindowTitle('你不是乎麻貓')
                msgBox.setText('你不是乎麻貓\n請正確填寫資料')
                msgBox.setStandardButtons(QMessageBox.Ok)
                msgBox.button(QMessageBox.Ok).hide()
                msgBox.setWindowFlag(QtCore.Qt.FramelessWindowHint)
                msgBox.button(QMessageBox.Ok).animateClick(1000)
                msgBox.exec_()  
                self.lineEdit.setText("????")
            elif self.lineEdit.text() == "????":
                msgBox = QMessageBox()
                msgBox.setWindowTitle('錯誤')
                msgBox.setText('????先生/小姐\n請正確填寫資料')
                msgBox.setStandardButtons(QMessageBox.Ok)
                msgBox.button(QMessageBox.Ok).hide()
                msgBox.setWindowFlag(QtCore.Qt.FramelessWindowHint)
                msgBox.button(QMessageBox.Ok).animateClick(1000)
                msgBox.exec_()  
                self.lineEdit.setText("????")
            else:
                ##upload data
                #face_id=self.image_num[self.pic_label]
                ntrid=self.sql_find_ntr_id(self.image_num[self.pic_label])
                self.sql_add_people(self.lineEdit.text(),ntrid)
                self.sql_remove_NTR(ntrid) 
                self.reloadData()
                ##upload data
                msgBox = QMessageBox()
                msgBox.setWindowTitle('成功')
                msgBox.setText('上傳成功')
                msgBox.setStandardButtons(QMessageBox.Ok)
                msgBox.button(QMessageBox.Ok).hide()
                msgBox.setWindowFlag(QtCore.Qt.FramelessWindowHint)
                msgBox.button(QMessageBox.Ok).animateClick(1000)
                msgBox.exec_()  
                self.lineEdit.setText("????")
    def next_img(self):
        print(str(self.pic_label)+"to "+str(self.pic_label+1))
        if self.pic_label+1 <=self.pic_maxlabel:
            self.pic_label+=1
            self.qtImshow(self.image[self.pic_label])
            self.lineEdit.setText("????")
        pass
    def priview_img(self):
        print(str(self.pic_label)+"to "+str(self.pic_label-1))
        if self.pic_label>0:
            self.pic_label-=1
            self.qtImshow(self.image[self.pic_label])
            self.lineEdit.setText("????")
        else:
            pass
    def actionConnect(self):
        self.Button_1.clicked.connect(lambda: self.setLineText("1"))
        self.Button_2.clicked.connect(lambda: self.setLineText("2"))
        self.Button_3.clicked.connect(lambda: self.setLineText("3"))
        self.Button_4.clicked.connect(lambda: self.setLineText("4"))
        self.Button_5.clicked.connect(lambda: self.setLineText("5"))
        self.Button_6.clicked.connect(lambda: self.setLineText("6"))
        self.Button_7.clicked.connect(lambda: self.setLineText("7"))
        self.Button_8.clicked.connect(lambda: self.setLineText("8"))
        self.Button_9.clicked.connect(lambda: self.setLineText("9"))
        self.Button_0.clicked.connect(lambda: self.setLineText("0"))
        self.Button_reload.clicked.connect(lambda: self.reloadData())
        self.Button_enter.clicked.connect(lambda: self.enter())
        self.Button_next.clicked.connect(lambda: self.next_img())
        self.Button_priview.clicked.connect(lambda: self.priview_img())
        self.Button_del.clicked.connect(lambda: self.lineEdit.setText(self.lineEdit.text()[:-1]if len(self.lineEdit.text()) > 1 and self.lineEdit.text()!="????" else "????"))

import img_rc

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.mousePressEvent=ui.mousePressEvent
    img=cv2.imread("unknown.jpg")
    ui.qtImshow(img)
    ui.setupImg()
    ui.actionConnect()
    MainWindow.showFullScreen()
    #MainWindow.show()
    sys.exit(app.exec_())

