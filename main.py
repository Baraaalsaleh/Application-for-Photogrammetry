from numpy.core.numeric import False_

import firebase_manager
from firebase_manager import *


#fire= firebase_manager.FirebaseMangager

#files =fire.listOfFiles(self)
#print('available files:')
#for f in files:
    #print(f.name)

#fire.download(fielpath='files/img.zip',giveAname='downloaded')
#fire.storage.child("files/myimages.zip").download(path= "files/myimages.zip",filename='down.zip')




import os

from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog, QPushButton
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import glob
import numpy as np
import zipfile
from process_image import *
import shutil
import sys
import subprocess
import threading
import meshing
from process_image import *
import time


FILE_PATH = os.path.dirname(__file__)
TEMP_DATA_DIR = os.path.join(FILE_PATH, "temp_data")
PROCESSED_IMAGES_DIR = os.path.join(FILE_PATH, "processed_images")
FireBase_IMAGES_DIR = os.path.join(FILE_PATH, "Downloaded")
APPLIED_CHANGES = [0, 100, 0, 100, 0, 255, 0, 255, 0, 255]
PROCESSED_IMAGES_PATHES = []
IMAGE_PATHES = []


'''cmd = "python process_image.py " + IMAGE_PATHES[self.image_index] + " " + path + " " + str(self.minX) + " " + str(self.maxX) + " " + str(self.minY) + " " + str(self.maxY) + " " + str(self.minR) + " " + str(self.maxR) + " " + str(self.minG) + " " + str(self.maxG) + " " + str(self.minB) + " " + str(self.maxB) + " " + str(downscale)
subprocess.call(cmd, shell=True)

x = threading.Thread(target= self.run_process_image(cmd), args=(1,))
x.start()'''


class Ui_MainWindow(QtWidgets.QMainWindow):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1045, 790)
        MainWindow.setStyleSheet("background-color: rgb(106, 106, 106);\n"
                                 "background-color: rgb(255, 255, 255);")
        self.minX = 0
        self.maxX = 100
        self.minY = 0
        self.maxY = 100
        self.minR = 0
        self.maxR = 255
        self.minG = 0
        self.maxG = 255
        self.minB = 0
        self.maxB = 255
        self.applied = False
        self.mesh_clicked = False
        self.speed_up_value = 1
        self.kmeans = None
        self.best_k = None
        self.check_boxes = []
        self.img_path = None
        self.unsup_ready = False
        self.sup_ready = False
        self.save_all = False
        self._translate = QtCore.QCoreApplication.translate
        self.window_size = (5, 5)
        self.model = None


        self.image_index = None
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1045, 790)
        MainWindow.setStyleSheet("background-color: rgb(106, 106, 106);\n"
                                 "background-color: rgb(255, 255, 255);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(8)
        sizePolicy.setVerticalStretch(5)
        sizePolicy.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy)
        self.tabWidget.setMouseTracking(True)
        self.tabWidget.setMovable(True)
        self.tabWidget.setObjectName("tabWidget")
        self.seite1 = QtWidgets.QWidget()
        self.seite1.setObjectName("seite1")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.seite1)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(20, 470, 741, 71))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.fotopath_seite1_linedit = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fotopath_seite1_linedit.sizePolicy().hasHeightForWidth())
        self.fotopath_seite1_linedit.setSizePolicy(sizePolicy)
        self.fotopath_seite1_linedit.setObjectName("fotopath_seite1_linedit")
        self.verticalLayout.addWidget(self.fotopath_seite1_linedit)
        self.verticalLayoutWidget_3 = QtWidgets.QWidget(self.seite1)
        self.verticalLayoutWidget_3.setGeometry(QtCore.QRect(799, 550, 161, 100))
        self.verticalLayoutWidget_3.setObjectName("verticalLayoutWidget_3")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_3)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.loschen_Button = QtWidgets.QPushButton(self.verticalLayoutWidget_3)
        self.loschen_Button.setStyleSheet("background-color: rgb(209, 255, 201);")
        self.loschen_Button.setObjectName("loschen_Button")
        self.verticalLayout_3.addWidget(self.loschen_Button)
        self.zeigen_Button = QtWidgets.QPushButton(self.verticalLayoutWidget_3)
        self.zeigen_Button.setStyleSheet("background-color: rgb(209, 255, 201);")
        self.zeigen_Button.setObjectName("zeigen_Button")
        self.verticalLayout_3.addWidget(self.zeigen_Button)
        self.baslik_label = QtWidgets.QLabel(self.seite1)
        self.baslik_label.setGeometry(QtCore.QRect(251, 100, 571, 61))
        self.baslik_label.setStyleSheet("\n""background-color: rgb(188, 223, 255);\n""font: 75 24pt \"Times New Roman\";")
        self.baslik_label.setObjectName("baslik_label")
        self.baslik_label.setAlignment(QtCore.Qt.AlignCenter)
        self.tau_logo = QtWidgets.QLabel(self.seite1)
        self.tau_logo.setGeometry(QtCore.QRect(10, 10, 301, 71))
        self.tau_logo.setText("")
        self.tau_logo.setPixmap(QtGui.QPixmap("../../Downloads/5566_4_th.png"))
        self.tau_logo.setScaledContents(True)
        self.tau_logo.setObjectName("tau_logo")
        self.tuberlin_logo = QtWidgets.QLabel(self.seite1)
        self.tuberlin_logo.setGeometry(QtCore.QRect(890, 20, 111, 61))
        self.tuberlin_logo.setText("")
        self.tuberlin_logo.setPixmap(QtGui.QPixmap("../../Downloads/WhatsApp Image 2021-04-26 at 01.53.53.jpeg"))
        self.tuberlin_logo.setScaledContents(True)
        self.tuberlin_logo.setObjectName("tuberlin_logo")
        self.verticalLayoutWidget_7 = QtWidgets.QWidget(self.seite1)
        self.verticalLayoutWidget_7.setGeometry(QtCore.QRect(20, 540, 741, 151))
        self.verticalLayoutWidget_7.setObjectName("verticalLayoutWidget_7")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_7)
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.listWidget_2 = QtWidgets.QListWidget(self.verticalLayoutWidget_7)
        self.listWidget_2.setObjectName("listWidget_2")
        self.verticalLayout_7.addWidget(self.listWidget_2)
        self.gridLayoutWidget_2 = QtWidgets.QWidget(self.seite1)
        self.gridLayoutWidget_2.setGeometry(QtCore.QRect(20, 180, 741, 301))
        self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.gridLayoutWidget_2)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.seite1foto_label = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.seite1foto_label.setText("")
        self.seite1foto_label.setObjectName("seite1foto_label")
        self.seite1foto_label.setScaledContents(True)
        self.seite1foto_label.setMaximumHeight(800)
        self.seite1foto_label.setMaximumWidth(600)
        self.gridLayout_3.addWidget(self.seite1foto_label, 0, 0, 1, 1)
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.seite1)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(770, 470, 241, 71))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.browsen_Button = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.browsen_Button.setStyleSheet("background-color: rgb(183, 221, 255);")
        self.browsen_Button.setObjectName("browsen_Button")
        self.horizontalLayout.addWidget(self.browsen_Button)
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.seite1)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(800, 360, 161, 101))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.firebase_Button = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.firebase_Button.setStyleSheet("background-color: rgb(255, 152, 168);")
        self.firebase_Button.setObjectName("firebase_Button")
        self.verticalLayout_2.addWidget(self.firebase_Button)
        self.seite1save_Button = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.seite1save_Button.setStyleSheet("background-color: rgb(255, 152, 168);")
        self.seite1save_Button.setObjectName("seite1save_Button")
        self.verticalLayout_2.addWidget(self.seite1save_Button)
        self.tabWidget.addTab(self.seite1, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.savechanges_button = QtWidgets.QPushButton(self.tab_2)
        self.savechanges_button.setGeometry(QtCore.QRect(840, 610, 151, 40))
        self.savechanges_button.setObjectName("savechanges_button")
        self.meshroom_button = QtWidgets.QPushButton(self.tab_2)
        self.meshroom_button.setObjectName("meshroom_button")
        self.meshroom_button.setGeometry(QtCore.QRect(840, 655, 151, 40))
        self.verticalLayoutWidget_5 = QtWidgets.QWidget(self.tab_2)
        self.verticalLayoutWidget_5.setGeometry(QtCore.QRect(830, 90, 31, 351))
        self.verticalLayoutWidget_5.setObjectName("verticalLayoutWidget_5")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_5)
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.verticalSlider = QtWidgets.QSlider(self.verticalLayoutWidget_5)
        self.verticalSlider.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.verticalSlider.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider.setTickInterval(0)
        self.verticalSlider.setObjectName("verticalSlider")
        self.verticalLayout_5.addWidget(self.verticalSlider)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_5.addItem(spacerItem)
        self.verticalSlider_2 = QtWidgets.QSlider(self.verticalLayoutWidget_5)
        self.verticalSlider_2.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider_2.setObjectName("verticalSlider_2")
        self.verticalLayout_5.addWidget(self.verticalSlider_2)
        self.verticalLayoutWidget_6 = QtWidgets.QWidget(self.tab_2)
        self.verticalLayoutWidget_6.setGeometry(QtCore.QRect(970, 90, 31, 351))
        self.verticalLayoutWidget_6.setObjectName("verticalLayoutWidget_6")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_6)
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.verticalSlider_3 = QtWidgets.QSlider(self.verticalLayoutWidget_6)
        self.verticalSlider_3.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider_3.setObjectName("verticalSlider_3")
        self.verticalLayout_6.addWidget(self.verticalSlider_3)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_6.addItem(spacerItem1)
        self.verticalSlider_4 = QtWidgets.QSlider(self.verticalLayoutWidget_6)
        self.verticalSlider_4.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider_4.setObjectName("verticalSlider_4")
        self.verticalLayout_6.addWidget(self.verticalSlider_4)
        self.gridLayoutWidget = QtWidgets.QWidget(self.tab_2)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(80, 90, 661, 411))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label.setText("")
        self.label.setObjectName("label")
        self.label.setMaximumHeight(800)
        self.label.setMaximumWidth(600)
        self.label.setScaledContents(True)
        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 1)
        self.verticalLayoutWidget_8 = QtWidgets.QWidget(self.tab_2)
        self.verticalLayoutWidget_8.setGeometry(QtCore.QRect(80, 640, 741, 61))
        self.verticalLayoutWidget_8.setObjectName("verticalLayoutWidget_8")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_8)
        self.verticalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.listWidget = QtWidgets.QListWidget(self.verticalLayoutWidget_8)
        self.listWidget.setObjectName("listWidget")
        self.verticalLayout_8.addWidget(self.listWidget)
        self.Xmin_label = QtWidgets.QLabel(self.tab_2)
        self.Xmin_label.setGeometry(QtCore.QRect(740, 140, 90, 20))
        self.Xmin_label.setAlignment(QtCore.Qt.AlignCenter)
        self.Xmin_label.setObjectName("Xmin_label")
        self.Ymin_label = QtWidgets.QLabel(self.tab_2)
        self.Ymin_label.setGeometry(QtCore.QRect(740, 370, 90, 20))
        self.Ymin_label.setObjectName("Ymin_label")
        self.Xmax_label = QtWidgets.QLabel(self.tab_2)
        self.Xmax_label.setGeometry(QtCore.QRect(880, 140, 90, 20))
        self.Xmax_label.setObjectName("Xmax_label")
        self.Ymax_label = QtWidgets.QLabel(self.tab_2)
        self.Ymax_label.setGeometry(QtCore.QRect(880, 370, 90, 20))
        self.Ymax_label.setObjectName("Ymax_label")
        self.label_2 = QtWidgets.QLabel(self.tab_2)
        self.label_2.setGeometry(QtCore.QRect(420,40, 211, 51))
        self.label_2.setStyleSheet("background-color: rgb(188, 223, 255);\n""font: 75 18pt \"Times New Roman\";")
        self.label_2.setObjectName("label_2")
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.red_low_Slider = QtWidgets.QSlider(self.tab_2)
        self.red_low_Slider.setGeometry(QtCore.QRect(260, 520, 191, 22))
        self.red_low_Slider.setOrientation(QtCore.Qt.Horizontal)
        self.red_low_Slider.setObjectName("red_low_Slider")
        self.green_low_Slider = QtWidgets.QSlider(self.tab_2)
        self.green_low_Slider.setGeometry(QtCore.QRect(260, 560, 191, 22))
        self.green_low_Slider.setOrientation(QtCore.Qt.Horizontal)
        self.green_low_Slider.setObjectName("green_low_Slider")
        self.red_up_Slider = QtWidgets.QSlider(self.tab_2)
        self.red_up_Slider.setGeometry(QtCore.QRect(630, 520, 181, 22))
        self.red_up_Slider.setOrientation(QtCore.Qt.Horizontal)
        self.red_up_Slider.setObjectName("red_up_Slider")
        self.green_up_Slider = QtWidgets.QSlider(self.tab_2)
        self.green_up_Slider.setGeometry(QtCore.QRect(630, 560, 181, 22))
        self.green_up_Slider.setOrientation(QtCore.Qt.Horizontal)
        self.green_up_Slider.setObjectName("green_up_Slider")
        self.blue_up_Slider = QtWidgets.QSlider(self.tab_2)
        self.blue_up_Slider.setGeometry(QtCore.QRect(630, 600, 181, 22))
        self.blue_up_Slider.setOrientation(QtCore.Qt.Horizontal)
        self.blue_up_Slider.setObjectName("blue_up_Slider")
        self.blue_low_Slider = QtWidgets.QSlider(self.tab_2)
        self.blue_low_Slider.setGeometry(QtCore.QRect(260, 600, 191, 22))
        self.blue_low_Slider.setOrientation(QtCore.Qt.Horizontal)
        self.blue_low_Slider.setObjectName("blue_low_Slider")
        self.anwenden_Button = QtWidgets.QPushButton(self.tab_2)
        self.anwenden_Button.setGeometry(QtCore.QRect(840, 475, 151, 40))
        self.anwenden_Button.setObjectName("anwenden_Button")
        self.label_3 = QtWidgets.QLabel(self.tab_2)
        self.label_3.setGeometry(QtCore.QRect(126, 520, 131, 20))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.tab_2)
        self.label_4.setGeometry(QtCore.QRect(110, 560, 141, 21))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.tab_2)
        self.label_5.setGeometry(QtCore.QRect(120, 600, 131, 20))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.tab_2)
        self.label_6.setGeometry(QtCore.QRect(490, 520, 131, 20))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.tab_2)
        self.label_7.setGeometry(QtCore.QRect(480, 560, 141, 20))
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.tab_2)
        self.label_8.setGeometry(QtCore.QRect(490, 600, 131, 20))
        self.label_8.setObjectName("label_8")
        self.tabWidget.addTab(self.tab_2, "")
        self.gridLayout.addWidget(self.tabWidget, 0, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.progress_label = QtWidgets.QLabel(self.tab_2)
        self.progress_label.setGeometry(QtCore.QRect(100, 650, 700, 20))
        self.progress_label.setAlignment(QtCore.Qt.AlignLeading)
        self.progress_label.setObjectName("progress_label")
        self.progress_label.setVisible(True)

        self.progress_bar = QtWidgets.QProgressBar(self.tab_2)
        self.progress_bar.setGeometry(QtCore.QRect(100, 675, 700, 10))
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)

        self.unsup = QtWidgets.QPushButton(self.tab_2)
        self.unsup.setGeometry(QtCore.QRect(840, 520, 160, 40))
        self.unsup.setObjectName("unsup")

        self.super = QtWidgets.QPushButton(self.tab_2)
        self.super.setGeometry(QtCore.QRect(840, 565, 151, 40))
        self.super.setObjectName("super")
        self.super.setEnabled(False)
        #self.super.setVisible(False)


        self.speed_up = QtWidgets.QSlider(self.tab_2)
        self.speed_up.setGeometry(QtCore.QRect(180, 675, 350, 20))
        self.speed_up.setOrientation(QtCore.Qt.Horizontal)
        self.speed_up.setObjectName("speed_up_slider")
        self.speed_up.setVisible(False)

        self.speed_up_label = QtWidgets.QLabel(self.tab_2)
        self.speed_up_label.setGeometry(QtCore.QRect(100, 650, 700, 20))
        self.speed_up_label.setObjectName("speed_up_label")
        self.speed_up_label.setVisible(False)

        self.speed_up_label2 = QtWidgets.QLabel(self.tab_2)
        self.speed_up_label2.setGeometry(QtCore.QRect(540, 675, 50, 22))
        self.speed_up_label2.setObjectName("speed_up_label2")
        self.speed_up_label2.setVisible(False)

        self.check_box_1 = QtWidgets.QCheckBox(self.tab_2)
        self.check_box_1.setGeometry(QtCore.QRect(100, 675, 68, 20))
        self.check_box_1.setObjectName("check_box_1")
        self.check_boxes.append(self.check_box_1)

        self.check_box_2 = QtWidgets.QCheckBox(self.tab_2)
        self.check_box_2.setGeometry(QtCore.QRect(170, 675, 68, 20))
        self.check_box_2.setObjectName("check_box_2")
        self.check_boxes.append(self.check_box_2)

        self.check_box_3 = QtWidgets.QCheckBox(self.tab_2)
        self.check_box_3.setGeometry(QtCore.QRect(240, 675, 68, 20))
        self.check_box_3.setObjectName("check_box_3")
        self.check_boxes.append(self.check_box_3)

        self.check_box_4 = QtWidgets.QCheckBox(self.tab_2)
        self.check_box_4.setGeometry(QtCore.QRect(310, 675, 68, 20))
        self.check_box_4.setObjectName("check_box_4")
        self.check_boxes.append(self.check_box_4)

        self.check_box_5 = QtWidgets.QCheckBox(self.tab_2)
        self.check_box_5.setGeometry(QtCore.QRect(380, 675, 68, 20))
        self.check_box_5.setObjectName("check_box_5")
        self.check_boxes.append(self.check_box_5)

        self.check_box_6 = QtWidgets.QCheckBox(self.tab_2)
        self.check_box_6.setGeometry(QtCore.QRect(468, 675, 68, 20))
        self.check_box_6.setObjectName("check_box_6")
        self.check_boxes.append(self.check_box_6)

        self.check_box_7 = QtWidgets.QCheckBox(self.tab_2)
        self.check_box_7.setGeometry(QtCore.QRect(520, 675, 68, 20))
        self.check_box_7.setObjectName("check_box_7")
        self.check_boxes.append(self.check_box_7)

        self.check_box_8 = QtWidgets.QCheckBox(self.tab_2)
        self.check_box_8.setGeometry(QtCore.QRect(590, 675, 68, 20))
        self.check_box_8.setObjectName("check_box_8")
        self.check_boxes.append(self.check_box_8)

        self.check_box_9 = QtWidgets.QCheckBox(self.tab_2)
        self.check_box_9.setGeometry(QtCore.QRect(660, 675, 68, 20))
        self.check_box_9.setObjectName("check_box_9")
        self.check_boxes.append(self.check_box_9)

        self.check_box_10 = QtWidgets.QCheckBox(self.tab_2)
        self.check_box_10.setGeometry(QtCore.QRect(730, 675, 68, 20))
        self.check_box_10.setObjectName("check_box_10")
        self.check_boxes.append(self.check_box_10)

        self.sift_check_box = QtWidgets.QCheckBox(self.tab_2)
        self.sift_check_box.setGeometry(QtCore.QRect(660, 675, 50, 20))
        self.sift_check_box.setObjectName("sift_check_box")
        self.sift_check_box.setChecked(True)
        self.sift_check_box.setVisible(False)

        self.akaze_check_box = QtWidgets.QCheckBox(self.tab_2)
        self.akaze_check_box.setGeometry(QtCore.QRect(730, 675, 50, 20))
        self.akaze_check_box.setObjectName("akaze_check_box")
        self.akaze_check_box.setChecked(True)
        self.akaze_check_box.setVisible(False)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)

        self.verticalSlider.setMinimum(0)
        self.verticalSlider.setMaximum(100)
        self.verticalSlider.setSingleStep(10)

        self.verticalSlider_2.setMinimum(0)
        self.verticalSlider_2.setMaximum(100)
        self.verticalSlider_2.setSingleStep(10)

        self.verticalSlider_3.setMinimum(0)
        self.verticalSlider_3.setMaximum(100)
        self.verticalSlider_3.setSingleStep(10)

        self.verticalSlider_4.setMinimum(0)
        self.verticalSlider_4.setMaximum(100)
        self.verticalSlider_4.setSingleStep(10)

        self.red_low_Slider.setMinimum(0)
        self.red_low_Slider.setMaximum(255)
        self.red_low_Slider.setSingleStep(10)

        self.red_up_Slider.setMinimum(0)
        self.red_up_Slider.setMaximum(255)
        self.red_up_Slider.setSingleStep(10)

        self.green_low_Slider.setMinimum(0)
        self.green_low_Slider.setMaximum(255)
        self.green_low_Slider.setSingleStep(10)

        self.green_up_Slider.setMinimum(0)
        self.green_up_Slider.setMaximum(255)
        self.green_up_Slider.setSingleStep(10)

        self.blue_low_Slider.setMinimum(0)
        self.blue_low_Slider.setMaximum(255)
        self.blue_low_Slider.setSingleStep(10)

        self.blue_up_Slider.setMinimum(0)
        self.blue_up_Slider.setMaximum(255)
        self.blue_up_Slider.setSingleStep(10)

        self.speed_up.setMinimum(1)
        self.speed_up.setMaximum(16)
        self.speed_up.setValue(1)

        self.verticalSlider_3.setValue(self.maxX)
        self.verticalSlider.setValue(self.minX)

        self.verticalSlider_4.setValue(self.maxY)
        self.verticalSlider_2.setValue(self.minY)

        self.red_low_Slider.setValue(self.minR)
        self.red_up_Slider.setValue(self.maxR)

        self.green_low_Slider.setValue(self.minG)
        self.green_up_Slider.setValue(self.maxG)

        self.blue_low_Slider.setValue(self.minB)
        self.blue_up_Slider.setValue(self.maxB)

        for cb in self.check_boxes:
            cb.setStyleSheet("background-color: rgb(188, 223, 255);\n""font: 75 6pt \"Times New Roman\";")
            cb.setChecked(True)
            cb.setVisible(False)


    def retranslateUi(self, MainWindow):
        
        MainWindow.setWindowTitle(self._translate("MainWindow", "TAU: Photogrammetry with Image Processing"))
        self.loschen_Button.setText(self._translate("MainWindow", "Delete Photo"))
        self.zeigen_Button.setText(self._translate("MainWindow", "Show Photo"))
        self.baslik_label.setText(self._translate("MainWindow", "Photogrammetry for 3D-Printers"))
        self.browsen_Button.setText(self._translate("MainWindow", "Import From Local Desk"))
        self.firebase_Button.setText(self._translate("MainWindow", "Choose from FireBase"))
        self.seite1save_Button.setText(self._translate("MainWindow", "Save"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.seite1), self._translate("MainWindow", "Import Images"))
        self.savechanges_button.setText(self._translate("MainWindow", "Save"))
        self.meshroom_button.setText(self._translate("MainWindow", "Mesh"))
        self.Xmin_label.setText(self._translate("MainWindow", "X Minimum: 0%"))
        self.Ymin_label.setText(self._translate("MainWindow", "Y Minimum: 0%"))
        self.Xmax_label.setText(self._translate("MainWindow", "X Maximum: 100%"))
        self.Ymax_label.setText(self._translate("MainWindow", "Y Maximum: 100%"))
        self.label_2.setText(self._translate("MainWindow", "Image Processing"))
        self.anwenden_Button.setText(self._translate("MainWindow", "Apply"))
        self.label_3.setText(self._translate("MainWindow", "Red Lower Threshold: 0"))
        self.label_4.setText(self._translate("MainWindow", "Green Lower Threshold: 0"))
        self.label_5.setText(self._translate("MainWindow", "Blue Lower Threshold: 0"))
        self.label_6.setText(self._translate("MainWindow", "Red Upper Threshold: 255"))
        self.label_7.setText(self._translate("MainWindow", "Green Upper Threshold: 255"))
        self.label_8.setText(self._translate("MainWindow", "Blue Upper Threshold: 255"))
        self.progress_label.setText(self._translate("MainWindow", "For reversing threshold or selection choose min value higher than the max value for the parameter you want to reverse"))
        self.unsup.setText(self._translate("MainWindow", "Use Unsupervised Learning\n to Segment Object"))
        self.super.setText(self._translate("MainWindow", "Use Supervised Learning\n to Segment Object"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), self._translate("MainWindow", "Image Processing and Meshing"))
        self.speed_up_label.setText(self._translate("MainWindow", "Choose Speeding Up Factor (it might reduce the quality of the result):"))
        self.speed_up_label2.setText(self._translate("MainWindow", "1X"))
        self.sift_check_box.setText(self._translate("MainWindow", "SIFT"))
        self.akaze_check_box.setText(self._translate("MainWindow", "AKAZE"))

        self.browsen_Button.clicked.connect(self.browsen)
        self.loschen_Button.clicked.connect(self.loschen)
        self.zeigen_Button.clicked.connect(self.zeigen)
        self.meshroom_button.clicked.connect(self.mesh)
        self.firebase_Button.clicked.connect(self.firebaselist)
        self.seite1save_Button.clicked.connect(self.save1)
        self.anwenden_Button.clicked.connect(lambda: self.crop_image(True))
        self.savechanges_button.clicked.connect(self.save_processed_image)
        self.unsup.clicked.connect(self.unsupervized)
        self.super.clicked.connect(self.supervised)

        

        self.verticalSlider.valueChanged[int].connect(self.x_min)
        self.verticalSlider_2.valueChanged[int].connect(self.y_min)
        self.verticalSlider_3.valueChanged[int].connect(self.x_max)
        self.verticalSlider_4.valueChanged[int].connect(self.y_max)

        self.red_low_Slider.valueChanged[int].connect(self.r_min)
        self.red_up_Slider.valueChanged[int].connect(self.r_max)

        self.green_low_Slider.valueChanged[int].connect(self.g_min)
        self.green_up_Slider.valueChanged[int].connect(self.g_max)

        self.blue_low_Slider.valueChanged[int].connect(self.b_min)
        self.blue_up_Slider.valueChanged[int].connect(self.b_max)

        self.speed_up.valueChanged[int].connect(self.speed)

        for cb in self.check_boxes:
            cb.stateChanged.connect(self.update_clusters)


    def browsen(self):
        self.fname = QFileDialog.getOpenFileNames(self, 'Open file')[0]
        self.imagePaths = self.fname
        for path in self.imagePaths:
            IMAGE_PATHES.append(path)
            PROCESSED_IMAGES_PATHES.append(path)
        self.pixmap = QPixmap(self.imagePaths)
        self.label.setPixmap(QPixmap(self.pixmap))
        self.label.resize(self.pixmap.width(), self.pixmap.height())
        self.listWidget_2.addItems(self.imagePaths)


    def loschen(self):
        index = self.listWidget_2.currentRow()
        if self.image_index:
            if self.image_index == index:
                self.seite1foto_label.setPixmap(QPixmap(None))
                self.label.setPixmap(QPixmap(None))
                self.image_index = None


        self.listWidget_2.takeItem(index)
        
        #IMAGE_PATHES.clear(self)
        IMAGE_PATHES.pop(index)
        PROCESSED_IMAGES_PATHES.pop(index)


    def zeigen(self):
        self.applied = False
        index = self.listWidget_2.currentRow()
        print(index)
        self.image_index = index
        print(self.image_index)
        item = PROCESSED_IMAGES_PATHES[index]
        self.pixmap = QPixmap(item)
        self.seite1foto_label.setPixmap(QPixmap(self.pixmap))
        self.seite1foto_label.resize(self.pixmap.width(), self.pixmap.height())
        self.label.setPixmap(QPixmap(self.pixmap))
        self.label.resize(self.pixmap.width(), self.pixmap.height())

        self.reset_sliders()
        self.hide_check_boxes()
        self.show_meshing_elements(False)
        self.reset_ai_elements()
        self.unsup_ready = False
        self.anwenden_Button.setText(self._translate("MainWindow", "Apply"))


    def mesh(self):
        
        self._translate = QtCore.QCoreApplication.translate

        if self.mesh_clicked and (self.sift_check_box.isChecked() or self.akaze_check_box.isChecked()):
            self.enable_sliders(False)
            self.progress_label.setText(self._translate("MainWindow", "Working on it ..."))
            self.show_progress_elements(True)
            self.show_meshing_elements(False)
            self.hide_check_boxes()

            aliceVision_path = "Meshroom-2021.1.0-win64\\Meshroom-2021.1.0\\aliceVision"
            print(self.speed_up_value)
            time.sleep(5)

            mesh(IMAGE_PATHES, PROCESSED_IMAGES_PATHES, self.progress_label, self.progress_bar, aliceVision_path, app, use_sift=self.sift_check_box.isChecked(), use_akaze=self.akaze_check_box.isChecked(), speed_up=self.speed_up_value)
            self.enable_sliders(True)
        
        elif self.mesh_clicked:
            self.progress_label.setText(self._translate("MainWindow", "One Descriptor at least should be choosen (SIFT or AKAZE)"))
        
        else:
            self.show_meshing_elements(True)
            self.hide_check_boxes()
            self.reset_ai_elements()


    def firebaselist(self):
        fire = FirebaseMangager()
        files = fire.listOfFiles()
        print('available files:')
        for f in files:
            self.listWidget_2.addItem(f.name)

    def save1(self):
        fire = FirebaseMangager()
        selected_item = self.listWidget_2.currentItem()
        if selected_item == None: return
        file_name = selected_item.text()
        fire.download(fielpath=file_name ,giveAname= FireBase_IMAGES_DIR + '/')
        with zipfile.ZipFile(FireBase_IMAGES_DIR + '/.zip', 'r') as zip_ref:
            zip_ref.extractall(FireBase_IMAGES_DIR + "/")

        list_files = glob.glob(FireBase_IMAGES_DIR + "/*.jpg")
        self.listWidget_2.clear()
        IMAGE_PATHES.clear()
        PROCESSED_IMAGES_PATHES.clear()
        self.listWidget_2.addItems(list_files)
        for item in list_files:
            item = item.replace('\\', '/')
            print(item)
            IMAGE_PATHES.append(item)
            PROCESSED_IMAGES_PATHES.append(item)

    def enable_sliders(self, state):
        self.speed_up.setEnabled(state)
        self.verticalSlider_3.setEnabled(state)
        self.verticalSlider.setEnabled(state)
        self.verticalSlider_4.setEnabled(state)
        self.verticalSlider_2.setEnabled(state)
        self.red_low_Slider.setEnabled(state)
        self.red_up_Slider.setEnabled(state)
        self.green_low_Slider.setEnabled(state)
        self.green_up_Slider.setEnabled(state)
        self.blue_low_Slider.setEnabled(state)
        self.blue_up_Slider.setEnabled(state)

    def hide_check_boxes(self):
        for cb in self.check_boxes:
            cb.setVisible(False)

    def show_progress_elements(self, state):
        self.progress_bar.setVisible(state)
        self.progress_bar.setValue(0)
    
    def show_meshing_elements(self, state):
        self.speed_up_label2.setVisible(state)
        self.speed_up_label.setVisible(state)
        self.speed_up.setVisible(state)
        self.sift_check_box.setVisible(state)
        self.akaze_check_box.setVisible(state)
        self.mesh_clicked = state
        
    
    def reset_ai_elements(self):
        self.kmeans = None
        self.best_k = None

    def x_min(self, value):

        self.minX = value

        self._translate = QtCore.QCoreApplication.translate
        self.Xmin_label.setText(self._translate("MainWindow", "X Minimum: " + str(self.minX) + "%"))

        print("x MIN >> " + str(self.minX))
        self.crop_image(use_original_size=False, force_redrawing = False)

    def x_max(self, value):
        
        self.maxX = value
        
        self._translate = QtCore.QCoreApplication.translate
        self.Xmax_label.setText(self._translate("MainWindow", "X Maximum: " + str(self.maxX) + "%"))

        print("x MAX >> " + str(self.maxX))
        self.crop_image(use_original_size=False, force_redrawing = False)

    def y_min(self, value):

        self.minY = value

        self._translate = QtCore.QCoreApplication.translate
        self.Ymin_label.setText(self._translate("MainWindow", "Y Minimum: " + str(self.minY) + "%"))
        
        print("y MIN >> " + str(self.minY))
        self.crop_image(use_original_size=False, force_redrawing = False)


    def y_max(self, value):

        self.maxY = value
        self._translate = QtCore.QCoreApplication.translate

        self.Ymax_label.setText(self._translate("MainWindow", "Y Maximum: " + str(self.maxY) + "%"))

        print("y MAX >> " + str(self.maxY))
        self.crop_image(use_original_size=False, force_redrawing = False)


    def r_min(self, value):
        self.minR = value

        self._translate = QtCore.QCoreApplication.translate

        self.label_3.setText(self._translate("MainWindow", "Red Lower Threshold: " + str(value)))

        print("Red MIN >> " + str(value))
        self.crop_image(use_original_size=False, force_redrawing = False)
    
    def r_max(self, value):
        self.maxR = value

        self._translate = QtCore.QCoreApplication.translate
        self.label_6.setText(self._translate("MainWindow", "Red Upper Threshold: " + str(value)))

        print("Red MAX >> " + str(value))
        self.crop_image(use_original_size=False, force_redrawing = False)

    def g_min(self, value):
        self.minG = value

        self._translate = QtCore.QCoreApplication.translate
        self.label_4.setText(self._translate("MainWindow", "Green Lower Threshold: " + str(value)))

        print("Green MIN >> " + str(value))
        self.crop_image(use_original_size=False, force_redrawing = False)
    
    def g_max(self, value):
        self.maxG = value

        self._translate = QtCore.QCoreApplication.translate
        self.label_7.setText(self._translate("MainWindow", "Green Upper Threshold: " + str(value)))

        print("Green MAX >> " + str(value))
        self.crop_image(use_original_size=False, force_redrawing = False)

    def b_min(self, value):
        self.minB = value

        self._translate = QtCore.QCoreApplication.translate
        self.label_5.setText(self._translate("MainWindow", "Blue Lower Threshold: " + str(value)))

        print("Blue MIN >> " + str(value))
        self.crop_image(use_original_size=False, force_redrawing = False)
    
    def b_max(self, value):
        self.maxB = value

        self._translate = QtCore.QCoreApplication.translate
        self.label_8.setText(self._translate("MainWindow", "Blue Upper Threshold: " + str(value)))

        print("Blue MAX >> " + str(value))
        self.crop_image(use_original_size=False, force_redrawing = False)
    
    def speed(self, value):
        self.speed_up_value = value

        self._translate = QtCore.QCoreApplication.translate
        self.speed_up_label2.setText(self._translate("MainWindow", str(self.speed_up_value) + "X"))

        print("SPEED UP VALUE >> " + str(self.speed_up_value))
        

    def crop_image(self, use_original_size = True, force_redrawing = True):
        print( "I am trying to process_image" )
        self.enable_sliders(False)
        self._translate = QtCore.QCoreApplication.translate

        self.progress_label.setText(self._translate("MainWindow", "Working on it ..."))
        self.show_progress_elements(True)
        print(1)
        if not self.unsup_ready and not self.sup_ready:

            if self.image_index != None:
                name = PROCESSED_IMAGES_PATHES[self.image_index].split('/')
                name = name[len(name)-1]
                
                path = TEMP_DATA_DIR + "/" + name
                img_path = ""

                if self.applied:
                    img_path = path
                else:
                    img_path = PROCESSED_IMAGES_PATHES[self.image_index]

                img = cv2.imread(img_path)
                print(use_original_size)

                if not use_original_size:
                    ratio = img.shape[0]/img.shape[1]
                    img = cv2.resize(img, (int(480), int(ratio*480)))
                    
                    path = path.replace(name, "Temp.jpg")

                else:
                    self.applied = True
                    print("self.applied is " + str(self.applied))

                y = img.shape[0]
                x = img.shape[1]
                y_min = int((self.minY / 100) * y)
                y_max = int((self.maxY / 100) * y)
                x_min = int((self.minX / 100) * x)
                x_max = int((self.maxX / 100) * x)


                new_img = process_image(img, 
                                        min_x=x_min, 
                                        max_x=x_max, 
                                        min_y=y_min, 
                                        max_y=y_max, 
                                        red_min=self.minR, 
                                        red_max=self.maxR, 
                                        green_min=self.minG, 
                                        green_max=self.maxG, 
                                        blue_min=self.minB, 
                                        blue_max=self.maxB,
                                        progress_label=self.progress_label,
                                        progress_bar=self.progress_bar,
                                        app = app,
                                        force_redraw=force_redrawing,
                                        mask = img)
                
                
                cv2.imwrite(path, new_img)
                print(2)

                self.pixmap = QPixmap(path)
                self.seite1foto_label.setPixmap(QPixmap(self.pixmap))
                self.seite1foto_label.resize(self.pixmap.width(), self.pixmap.height())
                self.label.setPixmap(QPixmap(self.pixmap))
                self.label.resize(self.pixmap.width(), self.pixmap.height())
                print( 3 )

                APPLIED_CHANGES = [self.minX, self.maxX, self.minY, self.maxY, self.minR, self.maxR, self.minG, self.maxG, self.minB, self.maxB]

        elif self.kmeans:

            self.hide_check_boxes()

            percentage = 100/len(PROCESSED_IMAGES_PATHES)

            for i in range(len(PROCESSED_IMAGES_PATHES)):
                self.progress_label.setText(self._translate("MainWindow", "Applying to image number " + str(i + 1) + " ..."))
                path = PROCESSED_IMAGES_PATHES[i]
                img = cv.imread(path)
                shape_1 = img.shape[0]*img.shape[1]
                x = img.reshape(shape_1, img.shape[2])
                labels = self.kmeans.predict(x)
                mask = (self.kmeans.cluster_centers_[labels]).reshape((img.shape[0], img.shape[1], img.shape[2]))
                mask = mask.astype(np.uint8)
                mask = cv.cvtColor(mask, cv.COLOR_RGB2BGR)

                centers = self.kmeans.cluster_centers_

                np_array = np.zeros(img.shape)
                y = img.shape[0]
                x = img.shape[1]

                for i in range(self.best_k):
                    if self.check_boxes[i].isChecked():
                        np_array += process_image(img, 
                                        min_x=0, 
                                        max_x=x, 
                                        min_y=0, 
                                        max_y=y, 
                                        red_min=centers[i][0] - 1, 
                                        red_max=centers[i][0] + 1, 
                                        green_min=centers[i][1] - 1, 
                                        green_max=centers[i][1] + 1, 
                                        blue_min=centers[i][2] - 1, 
                                        blue_max=centers[i][2] + 1,
                                        progress_label=self.progress_label,
                                        progress_bar=QtWidgets.QProgressBar(),
                                        app = app,
                                        force_redraw=False,
                                        mask= mask, 
                                        apply_white_back=False)
                
                img = np_array.astype(np.uint8)
                img = apply_white_background(img)
                for i in range(13, 3, -2):
                    kernel = np.ones((i,i),np.uint8)
                    img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
                
                name = path.split('/')
                name = name[len(name)-1]
                
                save_path = TEMP_DATA_DIR + "/" + name

                cv.imwrite(save_path, img)
                self.progress_bar.setValue(int(i*percentage))
                app.processEvents()

            self.save_all = True
            self.applied = True
        
        elif self.sup_ready and self.model is not None:
            i = 0
            to_process_images_pathes = self.get_processed_images()[2]
            slide_window(to_process_images_pathes, None, self.window_size, self.progress_bar, app, 0, 100, PROCESSED_IMAGES_DIR, None, use_original_size= True, model = self.model)
            self.save_all = True
            self.applied = True
            
                        
        self.progress_label.setText(self._translate("MainWindow", "Done"))
        self.show_progress_elements(False)
        self.enable_sliders(True)
        
    def get_processed_images(self):
        
        processed_images = []
        to_process_images = IMAGE_PATHES.copy()
        original_images = []

        for img_path, img_processed_path in zip(IMAGE_PATHES, PROCESSED_IMAGES_PATHES):
            if img_path != img_processed_path:
                processed_images.append(img_processed_path)
                original_images.append(img_path)
                to_process_images.remove(img_path)

        return (processed_images, original_images, to_process_images)

    def save_processed_image(self):
        self.enable_sliders(False)
        self._translate = QtCore.QCoreApplication.translate

        self.progress_label.setText(self._translate("MainWindow", "Working on it ..."))
        self.show_progress_elements(True)
        self.hide_check_boxes()
        self.show_meshing_elements(False)
        app.processEvents()
        time.sleep(0.5)

        if self.image_index != None and self.applied and not self.save_all:
            name = PROCESSED_IMAGES_PATHES[self.image_index].split('/')
            name = name[len(name)-1]
            
            img_path = TEMP_DATA_DIR + "/" + name
            target_path = PROCESSED_IMAGES_DIR + "/" + name

            shutil.copyfile(img_path, target_path)
            PROCESSED_IMAGES_PATHES[self.image_index] = target_path
            if len(self.get_processed_images()[0]) >= 2:
                self.super.setEnabled(True)
                self.unsup.setEnabled(False)
        
        elif self.applied and self.kmeans and self.save_all:
            for i in range(len(PROCESSED_IMAGES_PATHES)):
                path = PROCESSED_IMAGES_PATHES[i]
                name = path.split('/')
                name = name[len(name)-1]
                
                img_path = TEMP_DATA_DIR + "/" + name
                target_path = PROCESSED_IMAGES_DIR + "/" + name

                shutil.copyfile(img_path, target_path)
                PROCESSED_IMAGES_PATHES[i] = target_path
                self.super.setEnabled(False)
            
            self.save_all = False

        elif self.applied and self.model and self.save_all:
            for i in range(len(PROCESSED_IMAGES_PATHES)):
                path = PROCESSED_IMAGES_PATHES[i]
                name = path.split('/')
                name = name[len(name)-1]
                
                img_path = TEMP_DATA_DIR + "/" + name
                target_path = PROCESSED_IMAGES_DIR + "/" + name

                shutil.copyfile(img_path, target_path)
                PROCESSED_IMAGES_PATHES[i] = target_path
                self.unsup.setEnabled(False)
                
            
            self.save_all = False

        else:
            self.progress_label.setText(self._translate("MainWindow", "No image was choosen or you have not yet applied changes (press apply)"))
            app.processEvents()
            time.sleep(2)            
        
        self.progress_label.setText(self._translate("MainWindow", "Done"))
        self.progress_bar.setValue(100)
        app.processEvents()
        time.sleep(1)
        self.show_progress_elements(False)
        self.enable_sliders(True)
        
    def unsupervized(self):
        self._translate = QtCore.QCoreApplication.translate
        self.enable_sliders(False)
        self.show_progress_elements(True)
        self.hide_check_boxes()
        self.img_path, self.kmeans, self.best_k = use_kmeans(PROCESSED_IMAGES_PATHES, self.progress_bar, self.progress_label, app, TEMP_DATA_DIR, use_original_size=False)
        self.progress_label.setText(self._translate("MainWindow", "Choose clusters that you want to keep"))
        self.pixmap = QPixmap(self.img_path)
        self.seite1foto_label.setPixmap(QPixmap(self.pixmap))
        self.seite1foto_label.resize(self.pixmap.width(), self.pixmap.height())
        self.label.setPixmap(QPixmap(self.pixmap))
        self.label.resize(self.pixmap.width(), self.pixmap.height())
        self.update_clusters(True)
        centers = self.kmeans.cluster_centers_
        centers_np = np.array(centers).astype(int)

        self.show_progress_elements(False)
        for i in range(self.best_k):
            cb = self.check_boxes[i]
            cb.setChecked(True)
            cb.setVisible(True)
            cb.setStyleSheet("background-color: rgb(" + str(int(centers[i][2])) + ", " + str(int(centers[i][1])) + ", " + str(int(centers[i][0])) + ");\n""font: 75 6pt \"Times New Roman\";")
            cb.setText(str(np.flip(centers_np[i])))

        self.reset_sliders()
        self.enable_sliders(True)
        self.unsup_ready = True
        self.anwenden_Button.setText(self._translate("MainWindow", "Apply to all"))

    def update_clusters(self, value):
        if self.kmeans:
            centers = self.kmeans.cluster_centers_
            self.hide_check_boxes()
            print(centers)

            img = cv.imread(self.img_path)
            np_array = np.zeros(img.shape)
            y = img.shape[0]
            x = img.shape[1]

            for i in range(self.best_k):
                if self.check_boxes[i].isChecked():
                    np_array += process_image(img, 
                                    min_x=0, 
                                    max_x=x, 
                                    min_y=0, 
                                    max_y=y, 
                                    red_min=centers[i][0] - 1, 
                                    red_max=centers[i][0] + 1, 
                                    green_min=centers[i][1] - 1, 
                                    green_max=centers[i][1] + 1, 
                                    blue_min=centers[i][2] - 1, 
                                    blue_max=centers[i][2] + 1,
                                    progress_label=self.progress_label,
                                    progress_bar=self.progress_bar,
                                    app = app,
                                    force_redraw=False,
                                    mask = img,
                                    apply_white_back=False)
            
            img = np_array.astype(np.uint8)
            img = apply_white_background(img)
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            save_path = self.img_path.replace('png', 'jpg')
            cv.imwrite(save_path, img)
            self.pixmap = QPixmap(save_path)
            self.seite1foto_label.setPixmap(QPixmap(self.pixmap))
            self.seite1foto_label.resize(self.pixmap.width(), self.pixmap.height())
            self.label.setPixmap(QPixmap(self.pixmap))
            self.label.resize(self.pixmap.width(), self.pixmap.height())

            for i in range(self.best_k):
                self.check_boxes[i].setVisible(True)

            self._translate = QtCore.QCoreApplication.translate
            self.progress_label.setText(self._translate("MainWindow", "Choose clusters that you want to keep"))

    def supervised(self):

        self.show_progress_elements(True)

        pathes = self.get_processed_images()

        pathes, self.model = use_deep_learning(pathes[0], pathes[1], pathes[2], self.progress_bar, self.progress_label, app, TEMP_DATA_DIR, n_jobs = 4, window_size= self.window_size, use_original_size=False)
        
        print("Length of PATHES is " + str(len(pathes)))

        images = []
        for path in pathes:
            img = cv.imread(path)
            images.append(img)

        images = np.array(images)
        self.img_path = put_all_images_together(images, TEMP_DATA_DIR)

        self.pixmap = QPixmap(self.img_path)
        self.seite1foto_label.setPixmap(QPixmap(self.pixmap))
        self.seite1foto_label.resize(self.pixmap.width(), self.pixmap.height())
        self.label.setPixmap(QPixmap(self.pixmap))
        self.label.resize(self.pixmap.width(), self.pixmap.height())

        self.progress_label.setText(self._translate("MainWindow", "Click apply to use this model to all images (this may take a while)"))

        self.sup_ready = True

        
    def reset_sliders(self):

        self.minX = 0
        self.maxX = 100
        self.minY = 0
        self.maxY = 100
        self.minR = 0
        self.maxR = 255
        self.minG = 0
        self.maxG = 255
        self.minB = 0
        self.maxB = 255
        self.applied = False

        self.verticalSlider_3.setValue(self.maxX)
        self.verticalSlider.setValue(self.minX)

        self.verticalSlider_4.setValue(self.maxY)
        self.verticalSlider_2.setValue(self.minY)

        self.red_low_Slider.setValue(self.minR)
        self.red_up_Slider.setValue(self.maxR)

        self.green_low_Slider.setValue(self.minG)
        self.green_up_Slider.setValue(self.maxG)

        self.blue_low_Slider.setValue(self.minB)
        self.blue_up_Slider.setValue(self.maxB)


if __name__ == '__main__':

    shutil.rmtree(TEMP_DATA_DIR, ignore_errors=True)
    shutil.rmtree(PROCESSED_IMAGES_DIR, ignore_errors=True)
    shutil.rmtree(FireBase_IMAGES_DIR, ignore_errors=True)
    os.mkdir(TEMP_DATA_DIR)
    os.mkdir(PROCESSED_IMAGES_DIR)
    os.mkdir(FireBase_IMAGES_DIR)

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()

    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

