import cyMser
import source_window
import sys
import cv2
import visualize
import lifeCycle
from PyQt4 import QtGui

#I = cv2.imread("img.jpg")
#img = I.copy()
#I2 = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
#
#mser = cyMser.PyMser()
#
#a = mser.process_image(I2)
#
#visualize.draw_regions(img, mser)
#
#cv2.imshow("test", img)
#cv2.waitKey(0)

app = QtGui.QApplication(sys.argv)
w = source_window.Example()

sys.exit(app.exec_())