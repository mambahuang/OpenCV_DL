import os
import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout, QWidget
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtCore import QSize
from hw1_ui import Ui_Dialog

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
        self.setWindowTitle("HW1")
        self.relative_path = None
        # Load image button
        self.btn_load_image.clicked.connect(self.open_file_dialog)
        # Q1 button
        self.btn_color_seperate.clicked.connect(self.on_1_1_clicked)
        self.btn_color_transform.clicked.connect(self.on_1_2_clicked)
        self.btn_color_extraction.clicked.connect(self.on_1_3_clicked)
        # Q2 button
        self.btn_Gaussian_blur.clicked.connect(self.on_2_1_clicked)
        self.btn_Bilateral_filter.clicked.connect(self.on_2_2_clicked)
        self.btn_Median_filter.clicked.connect(self.on_2_3_clicked)
        # Q3 button
        self.btn_Sobel_x.clicked.connect(self.on_3_1_clicked)
        self.btn_Sobel_y.clicked.connect(self.on_3_2_clicked)
        self.btn_comb_Thre.clicked.connect(self.on_3_3_clicked)
        self.btn_Gradient_angle.clicked.connect(self.on_3_4_clicked)
        # Q4 button
        self.tx = None
        self.ty = None 
        self.angle = None 
        self.scale = None
        self.btn_Q4_transform.clicked.connect(self.operate)
        # Q4 lineEdit
        self.lineEdit.textChanged.connect(self.get_data)
        self.lineEdit_2.textChanged.connect(self.get_data)
        self.lineEdit_3.textChanged.connect(self.get_data)
        self.lineEdit_4.textChanged.connect(self.get_data)
        # Q5 
        self.btn_load_img.clicked.connect(self.load_img)
        self.btn_aug_img.clicked.connect(self.aug_img)
        self.btn_model_stru.clicked.connect(self.model_structure)
        self.btn_acc_loss.clicked.connect(self.acc_loss)
        self.btn_infer.clicked.connect(self.inference)
        self.lb_predicted.setText("Predicted: ")
        self.img_CIFAR10.setText("Inference Image")
        self.img_CIFAR10.setAlignment(QtCore.Qt.AlignCenter)
        self.filename = None
        self.classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.num_classes = 10
        self.model = torchvision.models.vgg19_bn(num_classes=self.num_classes)
        self.model = torch.load('vgg19_20231104_v3.pth', map_location=torch.device('cpu'))
        self.model.eval()

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
        print('===========Q1_1===========')
        img = cv2.imread(self.relative_path)
        height, width = img.shape[:2]
        black = np.zeros((height, width), np.uint8)
        cv2.imshow('BGR_img', img)
        b, g, r = cv2.split(img)
        cv2.imshow('B Channel', cv2.merge([b,black,black]))
        cv2.imshow('G Channel', cv2.merge([black, g, black]))
        cv2.imshow('R Channel', cv2.merge([black, black, r]))
        cv2.waitKey(0)
        print('========Q1_1 FINISH=======')
    # ==================== Q1_1 DONE =================

    # ==================== Q1_2 ======================
    def on_1_2_clicked(self):
        if self.relative_path == None:
            print("Please load image first!")
            return
        print('===========Q1_2===========')
        img = cv2.imread(self.relative_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('I1', gray) # gray image
        b, g, r = cv2.split(img)
        avg = 0.333*b + 0.333*g + 0.333*r
        cv2.imshow('I2', avg/255) # average image
        cv2.waitKey(0)
        print('========Q1_2 FINISH========')

    # ==================== Q1_2 DONE =================

    # ==================== Q1_3 ======================

    def on_1_3_clicked(self):
        if self.relative_path == None:
            print("Please load image first!")
            return
        print('===========Q1_3===========')
        img = cv2.imread(self.relative_path)
        cv2.imshow('BGR_img', img)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([15, 25, 25])
        higher = np.array([85, 255, 255])
        mask_yg = cv2.inRange(hsv, lower, higher)
        cv2.imshow('mask_yg', mask_yg)

        cv2.bitwise_not(img, img, mask_yg)
        cv2.imshow('I2', img)
        cv2.waitKey(0)
        print('========Q1_3 FINISH========')
    
    # ==================== Q1_3 DONE =================

    # ==================== Q2_1 ======================

    def gaussian_blur(self, x):
        img = cv2.imread(self.relative_path)
        img_blur = cv2.GaussianBlur(img, (2*x+1, 2*x+1), 0)
        cv2.imshow('Gaussian Blur', img_blur)
        

    def on_2_1_clicked(self):
        if self.relative_path == None:
            print("Please load image first!")
            return
        cv2.namedWindow('Gaussian Blur')
        cv2.createTrackbar('magnitude', 'Gaussian Blur', 1, 5, self.gaussian_blur)
        cv2.setTrackbarPos('magnitude', 'Gaussian Blur', 0)

    # ==================== Q2_1 DONE =================

    # ==================== Q2_2 ======================

    def bilateral_blur(self, x):
        img = cv2.imread(self.relative_path)
        img_blur = cv2.bilateralFilter(img, 2*(2*x+1), 90, 90)
        cv2.imshow('Bilateral Filter', img_blur)

    def on_2_2_clicked(self):
        if self.relative_path == None:
            print("Please load image first!")
            return
        cv2.namedWindow('Bilateral Filter')
        cv2.createTrackbar('magnitude', 'Bilateral Filter', 1, 5, self.bilateral_blur)
        cv2.setTrackbarPos('magnitude', 'Bilateral Filter', 0)

    # ==================== Q2_2 DONE ==================

    # ==================== Q2_3 =======================

    def median_blur(self, x):
        img = cv2.imread(self.relative_path)
        img_blur = cv2.medianBlur(img, 2*x+1)
        cv2.imshow('Median Filter', img_blur)

    def on_2_3_clicked(self):
        if self.relative_path == None:
            print("Please load image first!")
            return
        cv2.namedWindow('Median Filter')
        cv2.createTrackbar('magnitude', 'Median Filter', 1, 5, self.median_blur)
        cv2.setTrackbarPos('magnitude', 'Median Filter', 0)
    
    # ==================== Q2_3 DONE =================

    def Convolution2D(self, p_gaussian, kernel, padding = 1):

        # 宣告一個p_gaussian大小的0
        out = np.zeros(p_gaussian.shape)
        
        p_padding = np.zeros((p_gaussian.shape[0] + padding * 2, p_gaussian.shape[1] + padding * 2))

        
        p_padding[padding:-1*padding, padding:-1*padding] = p_gaussian

        for y in range(p_gaussian.shape[1]):
            for x in range(p_gaussian.shape[0]):
                out[x, y] = (kernel * p_padding[x: x + kernel.shape[0], y: y + kernel.shape[1]]).sum()
        return out

    # ==================== Q3_1 ====================

    def on_3_1_clicked(self):
        if self.relative_path == None:
            print("Please load image first!")
            return
        # Load the RGB image
        img = cv2.imread(self.relative_path)

        # Convert the RGB image to grayscale
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Define the Sobel X operator
        sobel_x_operator = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])

        # Apply Gaussian smoothing to the grayscale image (you can adjust the kernel size and standard deviation as needed)
        smoothed_image = cv2.GaussianBlur(gray_image, (3, 3), 0)

        # Apply the Sobel X operator (convolution)
        sobel_x = cv2.filter2D(smoothed_image, -1, sobel_x_operator)

        # Show the Sobel X result using cv2.imshow
        cv2.imshow('Sobel X Result', sobel_x)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
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

        # Define the Sobel Y operator for detecting horizontal edges
        sobel_y_operator = np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ])

        # Apply Gaussian smoothing to the grayscale image (you can adjust the kernel size and standard deviation as needed)
        smoothed_image = cv2.GaussianBlur(gray_image, (3, 3), 0)

        # Apply the Sobel Y operator (convolution)
        sobel_y = cv2.filter2D(smoothed_image, -1, sobel_y_operator)

        # Show the Sobel Y result using cv2.imshow
        cv2.imshow('Sobel Y Result', sobel_y)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ==================== Q3_2 DONE =================

    # ==================== Q3_3 ======================
    def comb_thres(self, gray_image):
        # Sobel Edge Detection 針對 X軸的垂直邊緣偵測
        sobel_x_operator = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])

        # Sobel Edge Detection 針對 y軸的水平邊緣偵測
        sobel_y_operator = np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ])

        # Apply Gaussian smoothing to the grayscale image (you can adjust the kernel size and standard deviation as needed)
        smoothed_image = cv2.GaussianBlur(gray_image, (3, 3), 0)

        # Apply the Sobel X operator (convolution)
        sobel_x = cv2.filter2D(smoothed_image, -1, sobel_x_operator)
        # Apply the Sobel Y operator (convolution)
        sobel_y = cv2.filter2D(smoothed_image, -1, sobel_y_operator)

        # comb = np.sqrt(x * x + y * y)
        combined = np.sqrt(sobel_x.astype(np.float64)**2 + sobel_y.astype(np.float64)**2)
        # Normalize the combination result to 0-255
        comb = (combined / np.max(combined) * 255).astype(np.uint8)
        # Given threshold
        threshold = 128

        # Apply thresholding
        thresholded_result = np.where(combined < threshold, 0, 255).astype(np.uint8)

        cv2.imshow('Combined Sobel', comb)
        cv2.imshow('Thresholded Result', thresholded_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def on_3_3_clicked(self):
        if self.relative_path == None:
            print("Please load image first!")
            return
        
        img = cv2.imread(self.relative_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.comb_thres(img_gray)

    # ==================== Q3_3 DONE =================

    # ==================== Q3_4 ======================

    def gradient_angle(self, gray_image):
        # Sobel Edge Detection 針對 X軸的垂直邊緣偵測
        sobel_x_operator = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])

        # Sobel Edge Detection 針對 y軸的水平邊緣偵測
        sobel_y_operator = np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ])
        # Apply Gaussian smoothing to the grayscale image (you can adjust the kernel size and standard deviation as needed)
        smoothed_image = cv2.GaussianBlur(gray_image, (3, 3), 0)

        # Apply the Sobel X operator (convolution)
        sobel_x = cv2.filter2D(smoothed_image, -1, sobel_x_operator)
        # Apply the Sobel Y operator (convolution)
        sobel_y = cv2.filter2D(smoothed_image, -1, sobel_y_operator)

        # Calculate the gradient angle
        gradient_angle = np.arctan2(sobel_y, sobel_x) * 180 / np.pi # Convert to degrees

        # print('sobel_x', sobel_x)
        # print('sobel_y', sobel_y)

        # Generate masks for specific angle ranges
        print('gradient_angle', gradient_angle)

        mask1 = ((gradient_angle >= 120) & (gradient_angle <= 180)).astype(np.uint8) * 255
        mask2 = ((gradient_angle >= 210) & (gradient_angle <= 330)).astype(np.uint8) * 255

        # comb = np.sqrt(x * x + y * y)
        combined = np.sqrt(sobel_x.astype(np.float64)**2 + sobel_y.astype(np.float64)**2)

        # Normalize the combination result to 0-255
        comb = (combined / np.max(combined) * 255).astype(np.uint8)

        # Apply the masks to the combined result
        result1 = cv2.bitwise_and(comb, comb, mask=mask1)
        result2 = cv2.bitwise_and(comb, comb, mask=mask2)

        # Show both results with cv2.imshow
        cv2.imshow('Mask 1 Result (120°-180°)', result1)
        cv2.imshow('Mask 2 Result (210°-330°)', result2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # x = self.Convolution2D(self.img_gaussian, kx)
        # y = self.Convolution2D(self.img_gaussian, ky)
        # grey_sobel_x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        # grey_sobel_y = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY)
        # gradient_angle = cv2.phase(grey_sobel_x, grey_sobel_y, angleInDegrees=True)
        # cv2.imshow('Gradient Angle', gradient_angle)
        # x_int16 = x.astype('int16')
        # y_int16 = y.astype('int16')
        # theta = np.zeros(x.shape, dtype='int16')
        # for i in range(x.shape[0]):
        #     for j in range(x.shape[1]):
        #         if x_int16[i][j] == 0 and y_int16[i][j] < 0:
        #             theta[i][j] = 270
        #         elif x_int16[i][j] == 0 and y_int16[i][j] > 0:
        #             theta[i][j] = 90
        #         elif x_int16[i][j] == 0 and y_int16[i][j] == 0:
        #             theta[i][j] = 0
        #         else:
        #             theta[i][j] = np.arctan(y_int16[i][j]/x_int16[i][j]) * 180 / np.pi
        #             theta[i][j] = (theta[i][j] + 360) % 360

        # mask1 = np.zeros(theta.shape, dtype='int16')
        # mask2 = np.zeros(theta.shape, dtype='int16')

        # for i in range(theta.shape[0]):
        #     for j in range(theta.shape[1]):
        #         # mask1
        #         if theta[i][j] >= 120 and theta[i][j] <= 180:
        #             mask1[i][j] = 255
        #         else:
        #             mask1[i][j] = 0
        #         # mask2
        #         if theta[i][j] >= 210 and theta[i][j] <= 330:
        #             mask2[i][j] = 255
        #         else:
        #             mask2[i][j] = 0

        # comb = np.sqrt(x * x + y * y)
        # comb = abs(comb).astype('uint8')
        # mask1 = abs(mask1).astype('uint8')
        # mask2 = abs(mask2).astype('uint8')

        # result = cv2.bitwise_and(comb, comb, mask1)
        # result2 = cv2.bitwise_and(comb, comb, mask2)

        # cv2.imshow('Gradient Angle', result)
        # cv2.imshow('Gradient Angle2', result2)

    def on_3_4_clicked(self):
        if self.relative_path == None:
            print("Please load image first!")
            return
        img = cv2.imread(self.relative_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.gradient_angle(img_gray)

    # ==================== Q3_4 DONE =================

    # ==================== Q4 ======================

    def get_data(self):
        try:
            self.angle = float(self.lineEdit.text())
            self.scale = float(self.lineEdit_2.text())
            self.tx = float(self.lineEdit_3.text())
            self.ty = float(self.lineEdit_4.text())
        except:
            pass

    def operate(self):
        if self.relative_path == None:
            print("Please load image first!")
            return
        if self.tx == None or self.ty == None or self.angle == None or self.scale == None:
            print("Please input data first!")
            return
        print('===========Q4_1===========')
        print('tx = ', self.tx)
        print('ty = ', self.ty)
        print('angle = ', self.angle)
        print('scale = ', self.scale)
        print('========Q4_1 FINISH========')
        self.lineEdit.setText('')
        self.lineEdit_2.setText('')
        self.lineEdit_3.setText('')
        self.lineEdit_4.setText('')

        img = cv2.imread(self.relative_path)
        # cv2.imshow('Original_burger', img)
        height, width = img.shape[:2]
        M = cv2.getRotationMatrix2D((240, 200), self.angle, self.scale)
        M[0, 2] += self.tx
        M[1, 2] += self.ty
        img_rotate = cv2.warpAffine(img, M, (width, height))
        cv2.imshow('After Shift', img_rotate)

    # ==================== Q4 DONE ==================

    # ==================== Q5 ======================

    # first button for loading image
    def load_img(self):
        self.filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Open a file', 
                                                         './', 
                                                         'All files(*.*)')
        print(self.filename[0])
        pixmap = QtGui.QPixmap(self.filename[0])
        pixmap = pixmap.scaled(128, 128, QtCore.Qt.KeepAspectRatio)
        self.img_CIFAR10.setText('')
        self.lb_predicted.setText("Predicted: ")
        label = QtWidgets.QLabel(self.img_CIFAR10)
        label.setPixmap(pixmap)
        label.show()

    # second button for augmenting image
    def aug_img(self):
        options = QtWidgets.QFileDialog.Options()
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select a Directory", 
                                                               './', options=options)
        if directory:
            current_folder = os.getcwd()
            relative_path = os.path.relpath(directory, current_folder)

        transform_v2 = v2.Compose([v2.ToImage(),
                                v2.ToDtype(torch.float32, scale=True),
                                v2.RandomHorizontalFlip(p=0.5),
                                v2.RandomVerticalFlip(p=0.5),
                                v2.RandomRotation(degrees=45)])
        
        png_files = os.listdir(relative_path)
        row, column = 3, 3  # subplot row and column

        fig, axs = plt.subplots(row, column, layout = 'constrained')

        for ax, png in zip(axs.flat, png_files):
            img = Image.open(relative_path + '/' + png)
            ax.imshow(transform_v2(img).numpy().transpose(1, 2, 0))
            ax.set_title(png.split('.')[0])

        plt.show()
    
    # Third button for training model
    def model_structure(self):
        summary(self.model, input_size=(3, 32, 32))

    # Forth button for accuracy and loss
    def acc_loss(self):
        options = QtWidgets.QFileDialog.Options()
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select a Directory", 
                                                               './', options=options)
        if directory:
            current_folder = os.getcwd()
            relative_path = os.path.relpath(directory, current_folder)
        
        # Load the two images
        image_loss = mpimg.imread(relative_path + '/loss.png')
        image_accu = mpimg.imread(relative_path + '/accuracy.png')

        # Create a figure and specify the number of rows and columns for the subplots
        plt.figure(figsize=(10, 5))  # Adjust the figure size as needed
        plt.subplot(1, 2, 1)  # 1 row, 2 columns, and select the first subplot
        plt.imshow(image_loss)
        plt.title('Image Loss')

        plt.subplot(1, 2, 2)  # 1 row, 2 columns, and select the second subplot
        plt.imshow(image_accu)
        plt.title('Image Accuracy')

        plt.show()

    # Fifth button for inference
    def inference(self):
        if self.filename == None:
            print("Please load image first!")
            return
        img = Image.open(self.filename[0]).convert('RGB')
        transform = transforms.Compose([transforms.ToTensor()])
        img = transform(img).unsqueeze(0)
        out_prob = self.model(img).detach()
        # print('out_prob: ', out_prob)

        softmax = torch.nn.Softmax(dim=1)
        softmax = softmax(out_prob).tolist()[0]
        # print('softmax: ', softmax)

        pred = torch.max(out_prob.data, 1)[1]
        pred = self.classes[pred.item()]
        # print('pred: ', pred)
        self.lb_predicted.setText(f"Predicted: {pred}")

        plt.figure(figsize=(10, 10))
        plt.bar([i for i in range(10)], softmax, align='center')
        plt.xticks(range(10), self.classes, fontsize=10, rotation=45)
        plt.yticks((0.2, 0.4, 0.6, 0.8, 1))
        plt.title('Probability of each class')
        plt.xlabel('Class')
        plt.ylabel('Probability')
        plt.show()

    # ==================== Q5 DONE ==================

        

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())