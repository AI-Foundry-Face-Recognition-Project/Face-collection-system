# -*- coding: utf-8 -*-

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
class Ui_MainWindow(object):
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
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

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
    def setLineText(self, text):
        if self.lineEdit.text() == "????":
            self.lineEdit.setText(text)
        else:
            self.lineEdit.setText(self.lineEdit.text() + text)
    def reloadData(self):
        self.lineEdit.setText("????")
    def enter(self):
        check = QMessageBox()
        check.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        reply =check.information(None, '上傳確認', '您好%s\n確定上傳嗎？' % self.lineEdit.text(), QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            ##uplode data
            msgBox = QMessageBox()
            msgBox.setWindowTitle('成功')
            msgBox.setText('上傳成功')
            msgBox.setStandardButtons(QMessageBox.Ok)
            msgBox.button(QMessageBox.Ok).hide()
            msgBox.setWindowFlag(QtCore.Qt.FramelessWindowHint)
            msgBox.button(QMessageBox.Ok).animateClick(1000)
            msgBox.exec_()  
            self.lineEdit.setText("????")
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
        self.Button_del.clicked.connect(lambda: self.lineEdit.setText(self.lineEdit.text()[:-1]if len(self.lineEdit.text()) > 1 and self.lineEdit.text()!="????" else "????"))

import img_rc

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    ui.actionConnect()
    MainWindow.showFullScreen()
    #MainWindow.show()
    sys.exit(app.exec_())

