# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'hw2.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1289, 1000)
        self.frame1 = QtWidgets.QFrame(Dialog)
        self.frame1.setGeometry(QtCore.QRect(200, 30, 291, 241))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.frame1.setFont(font)
        self.frame1.setWhatsThis("")
        self.frame1.setAccessibleName("")
        self.frame1.setAccessibleDescription("")
        self.frame1.setStyleSheet("")
        self.frame1.setFrameShape(QtWidgets.QFrame.Box)
        self.frame1.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame1.setLineWidth(1)
        self.frame1.setObjectName("frame1")
        self.label = QtWidgets.QLabel(self.frame1)
        self.label.setGeometry(QtCore.QRect(40, 10, 191, 41))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.btn_draw_contour = QtWidgets.QPushButton(self.frame1)
        self.btn_draw_contour.setGeometry(QtCore.QRect(60, 60, 161, 41))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.btn_draw_contour.setFont(font)
        self.btn_draw_contour.setObjectName("btn_draw_contour")
        self.btn_count_coins = QtWidgets.QPushButton(self.frame1)
        self.btn_count_coins.setGeometry(QtCore.QRect(60, 120, 161, 41))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.btn_count_coins.setFont(font)
        self.btn_count_coins.setObjectName("btn_count_coins")
        self.lb_num_coins = QtWidgets.QLabel(self.frame1)
        self.lb_num_coins.setGeometry(QtCore.QRect(10, 180, 271, 41))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.lb_num_coins.setFont(font)
        self.lb_num_coins.setAlignment(QtCore.Qt.AlignCenter)
        self.lb_num_coins.setObjectName("lb_num_coins")
        self.frame1_2 = QtWidgets.QFrame(Dialog)
        self.frame1_2.setGeometry(QtCore.QRect(200, 300, 291, 141))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.frame1_2.setFont(font)
        self.frame1_2.setWhatsThis("")
        self.frame1_2.setAccessibleName("")
        self.frame1_2.setAccessibleDescription("")
        self.frame1_2.setStyleSheet("")
        self.frame1_2.setFrameShape(QtWidgets.QFrame.Box)
        self.frame1_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame1_2.setLineWidth(1)
        self.frame1_2.setObjectName("frame1_2")
        self.label_2 = QtWidgets.QLabel(self.frame1_2)
        self.label_2.setGeometry(QtCore.QRect(40, 10, 191, 41))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.btn_hist_equ = QtWidgets.QPushButton(self.frame1_2)
        self.btn_hist_equ.setGeometry(QtCore.QRect(40, 60, 211, 41))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.btn_hist_equ.setFont(font)
        self.btn_hist_equ.setObjectName("btn_hist_equ")
        self.frame1_3 = QtWidgets.QFrame(Dialog)
        self.frame1_3.setGeometry(QtCore.QRect(200, 470, 291, 171))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.frame1_3.setFont(font)
        self.frame1_3.setWhatsThis("")
        self.frame1_3.setAccessibleName("")
        self.frame1_3.setAccessibleDescription("")
        self.frame1_3.setStyleSheet("")
        self.frame1_3.setFrameShape(QtWidgets.QFrame.Box)
        self.frame1_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame1_3.setLineWidth(1)
        self.frame1_3.setObjectName("frame1_3")
        self.btn_opening = QtWidgets.QPushButton(self.frame1_3)
        self.btn_opening.setGeometry(QtCore.QRect(10, 110, 271, 41))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.btn_opening.setFont(font)
        self.btn_opening.setObjectName("btn_opening")
        self.btn_closing = QtWidgets.QPushButton(self.frame1_3)
        self.btn_closing.setGeometry(QtCore.QRect(10, 60, 271, 41))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.btn_closing.setFont(font)
        self.btn_closing.setObjectName("btn_closing")
        self.label_3 = QtWidgets.QLabel(self.frame1_3)
        self.label_3.setGeometry(QtCore.QRect(30, 10, 211, 41))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.frame1_4 = QtWidgets.QFrame(Dialog)
        self.frame1_4.setGeometry(QtCore.QRect(540, 30, 671, 321))
        self.frame1_4.setFrameShape(QtWidgets.QFrame.Box)
        self.frame1_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame1_4.setObjectName("frame1_4")
        self.btn_Q4_show_stru = QtWidgets.QPushButton(self.frame1_4)
        self.btn_Q4_show_stru.setGeometry(QtCore.QRect(40, 70, 211, 31))
        self.btn_Q4_show_stru.setObjectName("btn_Q4_show_stru")
        self.label_4 = QtWidgets.QLabel(self.frame1_4)
        self.label_4.setGeometry(QtCore.QRect(10, 10, 241, 41))
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.btn_Q4_acc_loss = QtWidgets.QPushButton(self.frame1_4)
        self.btn_Q4_acc_loss.setGeometry(QtCore.QRect(40, 120, 211, 31))
        self.btn_Q4_acc_loss.setObjectName("btn_Q4_acc_loss")
        self.btn_Q4_predict = QtWidgets.QPushButton(self.frame1_4)
        self.btn_Q4_predict.setGeometry(QtCore.QRect(40, 170, 211, 31))
        self.btn_Q4_predict.setObjectName("btn_Q4_predict")
        self.btn_Q4_reset = QtWidgets.QPushButton(self.frame1_4)
        self.btn_Q4_reset.setGeometry(QtCore.QRect(40, 220, 211, 31))
        self.btn_Q4_reset.setObjectName("btn_Q4_reset")
        self.lb_Q4_img = QtWidgets.QLabel(self.frame1_4)
        self.lb_Q4_img.setGeometry(QtCore.QRect(274, 9, 391, 311))
        self.lb_Q4_img.setText("")
        self.lb_Q4_img.setObjectName("lb_Q4_img")
        self.lb_predict = QtWidgets.QLabel(self.frame1_4)
        self.lb_predict.setGeometry(QtCore.QRect(40, 270, 211, 41))
        self.lb_predict.setText("")
        self.lb_predict.setAlignment(QtCore.Qt.AlignCenter)
        self.lb_predict.setObjectName("lb_predict")
        self.frame1_5 = QtWidgets.QFrame(Dialog)
        self.frame1_5.setGeometry(QtCore.QRect(540, 380, 671, 481))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.frame1_5.setFont(font)
        self.frame1_5.setWhatsThis("")
        self.frame1_5.setAccessibleName("")
        self.frame1_5.setAccessibleDescription("")
        self.frame1_5.setStyleSheet("")
        self.frame1_5.setFrameShape(QtWidgets.QFrame.Box)
        self.frame1_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame1_5.setLineWidth(1)
        self.frame1_5.setObjectName("frame1_5")
        self.btn_infer = QtWidgets.QPushButton(self.frame1_5)
        self.btn_infer.setGeometry(QtCore.QRect(40, 300, 251, 41))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.btn_infer.setFont(font)
        self.btn_infer.setObjectName("btn_infer")
        self.btn_model_stru = QtWidgets.QPushButton(self.frame1_5)
        self.btn_model_stru.setGeometry(QtCore.QRect(40, 180, 251, 41))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.btn_model_stru.setFont(font)
        self.btn_model_stru.setObjectName("btn_model_stru")
        self.btn_load_img = QtWidgets.QPushButton(self.frame1_5)
        self.btn_load_img.setGeometry(QtCore.QRect(40, 60, 251, 41))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.btn_load_img.setFont(font)
        self.btn_load_img.setObjectName("btn_load_img")
        self.btn_comparision = QtWidgets.QPushButton(self.frame1_5)
        self.btn_comparision.setGeometry(QtCore.QRect(40, 240, 251, 41))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.btn_comparision.setFont(font)
        self.btn_comparision.setObjectName("btn_comparision")
        self.btn_img = QtWidgets.QPushButton(self.frame1_5)
        self.btn_img.setGeometry(QtCore.QRect(40, 120, 251, 41))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.btn_img.setFont(font)
        self.btn_img.setObjectName("btn_img")
        self.label_12 = QtWidgets.QLabel(self.frame1_5)
        self.label_12.setGeometry(QtCore.QRect(80, 10, 161, 41))
        self.label_12.setAlignment(QtCore.Qt.AlignCenter)
        self.label_12.setObjectName("label_12")
        self.lb_Q5_img = QtWidgets.QLabel(self.frame1_5)
        self.lb_Q5_img.setGeometry(QtCore.QRect(370, 90, 224, 224))
        self.lb_Q5_img.setText("")
        self.lb_Q5_img.setObjectName("lb_Q5_img")
        self.lb_Q5_predict = QtWidgets.QLabel(self.frame1_5)
        self.lb_Q5_predict.setGeometry(QtCore.QRect(310, 370, 301, 71))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lb_Q5_predict.setFont(font)
        self.lb_Q5_predict.setAlignment(QtCore.Qt.AlignCenter)
        self.lb_Q5_predict.setObjectName("lb_Q5_predict")
        self.btn_load_image = QtWidgets.QPushButton(Dialog)
        self.btn_load_image.setGeometry(QtCore.QRect(40, 120, 111, 51))
        self.btn_load_image.setObjectName("btn_load_image")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "1. Hough Circle Transform"))
        self.btn_draw_contour.setText(_translate("Dialog", "1.1 Draw Contour"))
        self.btn_count_coins.setText(_translate("Dialog", "1.2 Count Coins"))
        self.lb_num_coins.setText(_translate("Dialog", "There are _ coins in the image."))
        self.label_2.setText(_translate("Dialog", "2. Histogram Equalization"))
        self.btn_hist_equ.setText(_translate("Dialog", "2. Histogram equalization"))
        self.btn_opening.setText(_translate("Dialog", "3.2 Opening"))
        self.btn_closing.setText(_translate("Dialog", "3.1 Closing"))
        self.label_3.setText(_translate("Dialog", "3. Morphology Operation"))
        self.btn_Q4_show_stru.setText(_translate("Dialog", "1. Show Model Structure"))
        self.label_4.setText(_translate("Dialog", "4. MNIST Classifier Using VGG19"))
        self.btn_Q4_acc_loss.setText(_translate("Dialog", "2. Show Accuraccy and Loss"))
        self.btn_Q4_predict.setText(_translate("Dialog", "3. Predict"))
        self.btn_Q4_reset.setText(_translate("Dialog", "4. Reset"))
        self.btn_infer.setText(_translate("Dialog", "5.4. Inference"))
        self.btn_model_stru.setText(_translate("Dialog", "5.2 Show Model Structure"))
        self.btn_load_img.setText(_translate("Dialog", "Load Image"))
        self.btn_comparision.setText(_translate("Dialog", "5.3 Show Comparision"))
        self.btn_img.setText(_translate("Dialog", "5.1 Show Images"))
        self.label_12.setText(_translate("Dialog", "5. ResNet50"))
        self.lb_Q5_predict.setText(_translate("Dialog", "Predict:"))
        self.btn_load_image.setText(_translate("Dialog", "Load image"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
