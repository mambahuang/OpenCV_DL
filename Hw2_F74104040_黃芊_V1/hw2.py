import os
import sys
import cv2
import numpy as np
import random
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout, QWidget
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtCore import QSize
from PyQt5.QtGui import *
from hw2_ui import Ui_Dialog
from graffiti import GraffitiDialog
from train import ImageClassificationBase, DogsCatsCnnModelResNet50

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torchvision.models.vgg import VGG19_BN_Weights
from torchsummary import summary
import torchvision
import matplotlib.pyplot as plt

import os
from PIL import Image
import matplotlib.image as mpimg

class MainWindow(QMainWindow, Ui_Dialog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("HW2")
        self.relative_path = None
        # Load image button
        self.btn_load_image.clicked.connect(self.open_file_dialog)
        # Q1 button
        self.btn_draw_contour.clicked.connect(self.on_1_1_clicked)
        self.btn_count_coins.clicked.connect(self.on_1_2_clicked)
        # Q2 button
        self.btn_hist_equ.clicked.connect(self.on_2_clicked)
        # Q3 button
        self.btn_closing.clicked.connect(self.on_3_1_clicked)
        self.btn_opening.clicked.connect(self.on_3_2_clicked)
        # Q4 button
        self.btn_Q4_show_stru.clicked.connect(self.on_4_1_clicked)
        self.btn_Q4_acc_loss.clicked.connect(self.on_4_2_clicked)
        self.btn_Q4_predict.clicked.connect(self.on_4_3_clicked)
        self.btn_Q4_reset.clicked.connect(self.on_4_4_clicked)
        self.lb_Q4_img = GraffitiDialog(self.frame1_4)
        self.classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
        self.num_classes = 10
        self.modelq4 = torchvision.models.vgg19_bn(num_classes=self.num_classes)
        self.modelq4 = torch.load('./VGG19_model/vgg19_20231215_v3.pth', map_location=torch.device('cpu'))
        self.modelq4.eval()
        # Q5 
        self.btn_load_img.clicked.connect(self.load_img)
        self.btn_img.clicked.connect(self.show_img)
        self.btn_model_stru.clicked.connect(self.model_structure)
        self.btn_comparision.clicked.connect(self.comparision)
        self.btn_infer.clicked.connect(self.inference)
        self.lb_Q5_predict.setText("Predicted: ")
        self.lb_Q5_img.setText("Inference Image")
        self.lb_Q5_img.setAlignment(QtCore.Qt.AlignCenter)
        self.filename = None
        self.classesq5 = ['Cat', 'Dog']
        self.num_classes = 2
        self.model = torchvision.models.resnet50()
        self.model = torch.load('./ResNet50_model/ResNet_20231218.pth',
                                 map_location=torch.device('cpu'))
        # ResNet50 without erasing
        device = torch.device('cpu')
        self.model_r_50 = DogsCatsCnnModelResNet50()
        self.model_r_50.load_state_dict(torch.load('./ResNet50_model/ResNet_20231221.pth', 
                                                   map_location=device))
        self.model_r_50.eval()
        # ResNet50 with erasing
        self.model_r_50_e = DogsCatsCnnModelResNet50()
        self.model_r_50_e.load_state_dict(torch.load('./ResNet50_model/ResNet_20231221_erased.pth', 
                                                     map_location=device))
        self.model_r_50_e.eval()

    # Function for loading image
    def open_file_dialog(self):
        options = QtWidgets.QFileDialog.Options()
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open a file', 
                                                             './', 'All files (*.*)', 
                                                             options=options)
        
        if file_path:
            # Get the current working directory
            current_directory = os.getcwd()
            
            # Calculate the relative path from the current working directory
            self.relative_path = os.path.relpath(file_path, current_directory)
            
            print("Selected file path:", file_path)
            print("Relative path:", self.relative_path)
        
    # ==================== Q1_1 ======================
    def on_1_1_clicked(self):
        if self.relative_path == None:
            print("Please load image first!")
            return
        src = cv2.imread(self.relative_path)
        img = cv2.imread(self.relative_path)
        black = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grey = cv2.GaussianBlur(grey, (5, 5), 0)
        
        circles = cv2.HoughCircles(grey, cv2.HOUGH_GRADIENT, 
                                   1, 20, param1=90, param2=15, minRadius=15, maxRadius=20)
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(black, (i[0], i[1]), 2, (255, 255, 255), 2)

        plt.figure(figsize=(10, 2))
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(src)
        plt.title('Original')
        # img_processed
        plt.subplot(1, 3, 2)
        plt.imshow(img)
        plt.title('img_processed')
        # Circle_Center
        plt.subplot(1, 3, 3)
        plt.imshow(black)
        plt.title('Circle_Center')
        plt.show()

        # cv2.imshow("Original", src)
        # cv2.imshow("img_processed", img)
        # cv2.imshow("Circle_Center", black)
        cv2.waitKey(0)
    # ==================== Q1_1 DONE =================

    # ==================== Q1_2 ======================
    def on_1_2_clicked(self):
        if self.relative_path == None:
            print("Please load image first!")
            return
        img = cv2.imread(self.relative_path)
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grey = cv2.GaussianBlur(grey, (5, 5), 0)
        
        circles = cv2.HoughCircles(grey, cv2.HOUGH_GRADIENT, 
                                   1, 20, param1=90, param2=15, minRadius=15, maxRadius=20)
        circles = np.uint16(np.around(circles))
        print("Number of coins:", len(circles[0, :]))
        self.lb_num_coins.setText(f"There are {len(circles[0, :])} coins in the image.")

    # ==================== Q1_2 DONE =================

    # ==================== Q2 ========================   

    def on_2_clicked(self):
        if self.relative_path == None:
            print("Please load image first!")
            return
        # Load the image
        num_list = range(256)
        img = cv2.imread(self.relative_path)
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        # Convert the image to grayscale
        grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Apply histogram equalization
        equ = cv2.equalizeHist(grey)
        hist_euq = cv2.calcHist([equ], [0], None, [256], [0, 256])

        plt.figure(figsize=(20, 10))
        # Original image
        plt.subplot(2, 3, 1)
        plt.imshow(img)
        plt.title('Original Image')
        plt.subplot(2, 3, 4)
        plt.bar(num_list, hist.flatten())
        plt.title('Histogram of Original')
        plt.xlabel('Grey Scale')
        plt.ylabel('Frequency')
        # Equalize with OpenCV
        plt.subplot(2, 3, 2)
        plt.imshow(equ, cmap='gray')
        plt.title('Equalize with OpenCV')
        plt.subplot(2, 3, 5)
        # plt.hist(equ.flatten(), 256, [0, 256], color='b')
        plt.bar(num_list, hist_euq.flatten())
        plt.title('Histogram of Equalize with OpenCV')
        plt.xlabel('Grey Scale')
        plt.ylabel('Frequency')
        # Equalize with my function
        equ_my = self.HistogramEqualization(grey)
        my_hist_euq = cv2.calcHist([equ_my], [0], None, [256], [0, 256])
        plt.subplot(2, 3, 3)
        plt.imshow(equ_my, cmap='gray')
        plt.title('Equalized Manually')
        plt.subplot(2, 3, 6)
        # plt.hist(equ_my.flatten(), 256, [0, 256], color='b')
        plt.bar(num_list, my_hist_euq.flatten())
        plt.title('Histogram of Equalized (Manual)')
        plt.xlabel('Grey Scale')
        plt.ylabel('Frequency')
        plt.show()

    def HistogramEqualization(self, img):
        # Get the height and width of the image
        height, width = img.shape
        # Calculate the number of pixels
        num_pixels = height * width
        # Calculate the histogram of the image
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        # Calculate the cumulative histogram
        cum_hist = np.cumsum(hist)
        # Calculate the cumulative histogram
        cum_hist = cum_hist / num_pixels
        # Calculate the cumulative histogram
        cum_hist = cum_hist * 255
        # Calculate the cumulative histogram
        cum_hist = np.uint8(cum_hist)
        # Calculate the cumulative histogram
        equ = cum_hist[img]
        return equ
    # ==================== Q2_1 DONE =================

    # ==================== Q3_1 ====================

    def on_3_1_clicked(self):
        if self.relative_path == None:
            print("Please load image first!")
            return
        # Load the RGB image
        img = cv2.imread(self.relative_path)

        # Convert the RGB image to grayscale
        grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary_img = cv2.threshold(grey_image, 127, 255, cv2.THRESH_BINARY)[1]
        
        kernel = np.ones((3, 3), np.uint8)
        # Apply closing
        cls = self.closing(binary_img, kernel)
        # self.test_closing(binary_img, kernel)
        cv2.imshow("closing", cls)
    
    def closing(self, binary_img, kernel):
        w = binary_img.shape[0]
        h = binary_img.shape[1]
        dilation_img = np.zeros((w, h), np.uint8)
        erosion_img = np.ones((w, h), np.uint8)
        erosion_img = erosion_img * 255
        
        # dilation
        for i in range(1, w - 1):
            for j in range(1, h - 1):
                if binary_img[i - 1, j - 1] == 255 or binary_img[i - 1, j] == 255 or binary_img[i - 1, j + 1] == 255 or binary_img[i, j - 1] == 255 or binary_img[i, j] == 255 or binary_img[i, j + 1] == 255 or binary_img[i + 1, j - 1] == 255 or binary_img[i + 1, j] == 255 or binary_img[i + 1, j + 1] == 255:
                    dilation_img[i, j] = 255
        # cv2.imshow("dilation", dilation_img)

        # erosion
        for i in range(1, w - 1):
            for j in range(1, h - 1):
                if dilation_img[i - 1, j - 1] == 0 or dilation_img[i - 1, j] == 0 or dilation_img[i - 1, j + 1] == 0 or dilation_img[i, j - 1] == 0 or dilation_img[i, j] == 0 or dilation_img[i, j + 1] == 0 or dilation_img[i + 1, j - 1] == 0 or dilation_img[i + 1, j] == 0 or dilation_img[i + 1, j + 1] == 0:
                    erosion_img[i, j] = 0
        # cv2.imshow("erosion", erosion_img)
        
        return erosion_img
    
    def test_closing(self, binary_img, kernel):
        cv2.dilate(binary_img, kernel, binary_img, iterations=1)
        cv2.erode(binary_img, kernel, binary_img, iterations=1)
        cv2.imshow("test", binary_img)
    
    # ==================== Q3_1 DONE ===============

    # ==================== Q3_2 ====================

    def on_3_2_clicked(self):
        if self.relative_path == None:
            print("Please load image first!")
            return
        
        # Load the RGB image
        img = cv2.imread(self.relative_path)

        # Convert the RGB image to grayscale
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary_img = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)[1]
        
        kernel = np.ones((3, 3), np.uint8)
        # Apply opening
        op = self.opening(binary_img, kernel)
        cv2.imshow("opening", op)
        # self.test_opening(binary_img, kernel)

    def opening(self, binary_img, kernel):
        w = binary_img.shape[0]
        h = binary_img.shape[1]
        dilation_img = np.zeros((w, h), np.uint8)
        erosion_img = np.zeros((w, h), np.uint8)
        erosion_img = erosion_img * 255
        
        # erosion
        for i in range(1, w - 1):
            for j in range(1, h - 1):
                if binary_img[i - 1, j - 1] == 0 or binary_img[i - 1, j] == 0 or binary_img[i - 1, j + 1] == 0 or binary_img[i, j - 1] == 0 or binary_img[i, j] == 0 or binary_img[i, j + 1] == 0 or binary_img[i + 1, j - 1] == 0 or binary_img[i + 1, j] == 0 or binary_img[i + 1, j + 1] == 0:
                    erosion_img[i, j] = 0
                else:
                    erosion_img[i, j] = 255
        cv2.imshow("erosion", erosion_img)

        # dilation
        for i in range(1, w - 1):
            for j in range(1, h - 1):
                if erosion_img[i - 1, j - 1] == 255 or erosion_img[i - 1, j] == 255 or erosion_img[i - 1, j + 1] == 255 or erosion_img[i, j - 1] == 255 or erosion_img[i, j] == 255 or erosion_img[i, j + 1] == 255 or erosion_img[i + 1, j - 1] == 255 or erosion_img[i + 1, j] == 255 or erosion_img[i + 1, j + 1] == 255:
                    dilation_img[i, j] = 255
        cv2.imshow("dilation", dilation_img)
        
        return dilation_img
        
    def test_opening(self, img, kernel):
        cv2.erode(img, kernel, img, iterations=1)
        cv2.dilate(img, kernel, img, iterations=1)
        cv2.imshow("test", img)

    # ================= Q3_2 DONE ==================

    # ==================== Q4 ======================
    
    # Show the structure of VGG19
    def on_4_1_clicked(self):
        summary(self.modelq4, input_size=(3, 32, 32))
    
    def on_4_2_clicked(self):
        GraffitiDialog.showImage(self.lb_Q4_img)

    def on_4_3_clicked(self):
        GraffitiDialog.save(self.lb_Q4_img)
        self.predict()

    def predict(self):

        img = Image.open("graffiti.png").convert('RGB')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Normalize((0.5, ), (0.5, ))
        ])

        img = transform(img).unsqueeze(0)

        with torch.no_grad():
            out_prob = self.modelq4(img)
        # print('out_prob: ', out_prob)

        softmax = torch.nn.Softmax(dim=1)
        softmax = softmax(out_prob).tolist()[0]
        # print('softmax: ', softmax)

        pred = torch.max(out_prob.data, 1)[1]
        pred = self.classes[pred.item()]
        # print('pred: ', pred)
        self.lb_predict.setText(pred)

        plt.figure(figsize=(10, 10))
        plt.bar([i for i in range(10)], softmax, align='center')
        plt.xticks(range(10), self.classes, fontsize=12)
        plt.yticks((0.2, 0.4, 0.6, 0.8, 1))
        plt.title('Probability of each class')
        plt.xlabel('Class')
        plt.ylabel('Probability')
        plt.show()

    def on_4_4_clicked(self):
        GraffitiDialog.clear(self.lb_Q4_img)
        self.lb_predict.setText("")
        

    # ==================== Q4 DONE ==================

    # ==================== Q5 ======================

    # first button for loading image
    def load_img(self):
        self.filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Open a file', 
                                                         './', 
                                                         'All files(*.*)')
        print(self.filename[0])
        pixmap = QtGui.QPixmap(self.filename[0])
        pixmap = pixmap.scaled(224, 224)
        self.lb_Q5_img.setText('')
        self.lb_Q5_predict.setText("Predicted: ")
        label = QtWidgets.QLabel(self.lb_Q5_img)
        label.setPixmap(pixmap)
        label.show()

    # second button for showing cat, dog images
    def show_img(self):
        options = QtWidgets.QFileDialog.Options()
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select a Directory", 
                                                               './', options=options)
        if directory:
            current_folder = os.getcwd()
            relative_path = os.path.relpath(directory, current_folder)
        
        if relative_path == None:
            print("Please load image first!")
            return
        # Path : ./inference dataset/Cat and inference dataset/Dog
        data_folder = os.listdir(relative_path)
        # ./inference dataset/Cat and ./inference dataset/Dog
        png_cat = os.listdir(relative_path + '/' + data_folder[0])
        png_dog = os.listdir(relative_path + '/' + data_folder[1])
        
        figure = plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        img = Image.open(relative_path + '/' + data_folder[0] + '/' + png_cat[random.randint(0, len(png_cat) - 1)])
        img = img.resize((224, 224), Image.LANCZOS)
        plt.imshow(img)
        plt.title('Cat')

        plt.subplot(1, 2, 2)
        img = Image.open(relative_path + '/' + data_folder[1] + '/' + png_dog[random.randint(0, len(png_dog) - 1)])
        img = img.resize((224, 224), Image.LANCZOS)
        plt.imshow(img)
        plt.title('Dog')

        plt.show()
    
    # Third button for model structure
    def model_structure(self):
        summary(self.model, input_size=(3, 224, 224))

    # Forth button for comparing erasing and non-erasing images
    def comparision(self):
        filename = './other_images/comparison.png'
        img = Image.open(filename)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    
    # Fifth button for inference
    def inference(self):
        if self.filename == None:
            print("Please load image first!")
            return
        img = Image.open(self.filename[0]).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.CenterCrop(224),
            transforms.ToTensor()])
        img = transform(img).unsqueeze(0)
        out_prob = self.model_r_50_e(img).detach()
        # print('out_prob: ', out_prob)

        pred = torch.max(out_prob.data, dim=1)[1]
        pred = self.classesq5[pred.item()]
        # print('pred: ', pred)
        self.lb_Q5_predict.setText(f"Predicted: {pred}")

    # ==================== Q5 DONE ==================


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())