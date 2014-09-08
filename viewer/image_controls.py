__author__ = 'flipajs'

import cv2
from gui.img_controls import img_controls
import sys
from PyQt4 import QtGui
import video_manager


def video_example():
    video = video_manager.VideoManager('../../data/eight/eight.m4v')

    # self.capture = cv2.cv.CreateFileCapture('src/eight.m4v')
    #
    # cv2.cv.SetCaptureProperty(self.capture, cv2.cv.CV_CAP_PROP_POS_FRAMES, 100)
    #
    # f, img = self.capture.read()

    img = video.next_img()
    cv2.imshow("test", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    ex = img_controls.ImgControls()

    app.exec_()
    app.deleteLater()
    sys.exit()