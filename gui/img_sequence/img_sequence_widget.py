__author__ = 'flipajs'
from PyQt4 import QtGui, QtCore
import sys
from viewer import video_manager
import cv2
import ImageQt
import numpy as np

class ImgSequenceWidget(QtGui.QWidget):
    def __init__(self, video_manager):
        super(ImgSequenceWidget, self).__init__()
        self.video = video_manager
        self.crop_width = 100
        self.crop_height = 100

        self.verticalLayoutWidget_2 = QtGui.QWidget()
        self.scrollArea_2 = QtGui.QScrollArea()
        self.scrollArea_2.setWidgetResizable(True)

        self.scrollAreaWidgetContents_2 = QtGui.QWidget()

        self.verticalLayout_2 = QtGui.QVBoxLayout(self.scrollAreaWidgetContents_2)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setMargin(0)

        self.grid = QtGui.QGridLayout()
        self.grid.setSizeConstraint(QtGui.QLayout.SetNoConstraint)
        self.grid.setSpacing(1)

        self.verticalLayout_2.addLayout(self.grid)
        self.scrollArea_2.setWidget(self.scrollAreaWidgetContents_2)

        self.main_layout = QtGui.QVBoxLayout()
        self.setLayout(self.main_layout)
        self.main_layout.addWidget(self.scrollArea_2)

        # self.setLayout(self.grid)

        # img = self.video.seek_frame(100)
        self.update()
        self.show()

    def update_sequence(self, frame, length, width=-1, height=-1):
        return
        gui = QtGui.QApplication.processEvents

        i = 1
        j = 1
        img = self.video.seek_frame(frame)
        for f in range(frame, frame+length):
            crop = img[450:850, 600:1000, :].copy()

            img_q = ImageQt.QImage(crop.data, crop.shape[1], crop.shape[0], crop.shape[1]*3, 13)
            pix_map = QtGui.QPixmap.fromImage(img_q.rgbSwapped())

            item = QtGui.QLabel()

            item.setScaledContents(True)
            item.setFixedWidth(pix_map.width()/4)
            item.setFixedHeight(pix_map.height()/4)
            item.setPixmap(pix_map)

            self.grid.addWidget(item, j, i)
            i += 1
            if i % 4 == 0:
                i = 1
                j += 1

            if j < 10:
                gui()

            img = self.video.next_img()


if __name__ == "__main__":
    print "TEST"
    # app = QtGui.QApplication(sys.argv)
    #
    # vid = video_manager.VideoManager('/home/flipajs/my_video-16_c.mkv')
    # ex = ImgSequenceWidget(vid)
    # ex.update_sequence(100, 500)
    #
    # app.exec_()
    # app.deleteLater()
    # sys.exit()