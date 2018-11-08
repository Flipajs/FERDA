from __future__ import print_function
from __future__ import unicode_literals
import math

# # -*- coding: utf-8 -*-
# #!/usr/bin/env python
#
# import sys
# from PyQt4.QtCore import *
# from PyQt4.QtGui import *
# from PyQt4.QtWebKit import *
#
# class MainWindow(QMainWindow):
#     def __init__(self, parent = None):
#         QMainWindow.__init__(self, parent)
#         self.resize(700, 250)
#         self.btn = QPushButton(self)
#         self.btn.setText("Test")
#         QObject.connect(self.btn, SIGNAL("clicked()"), self.btnClicked)
#         self.txt = QTextEdit(self)
#         self.txt.move(0, 30)
#         self.txt.resize(700, 200)
#         self.alist = [["ls", ""], ["uname", ["--processor"]], ["uname", ['--machine']]]
#
#     def btnClicked(self):
#         self.process = QProcess(self)
#         QObject.connect(self.process, SIGNAL("finished(int)"), self.finished)
#         QObject.connect(self.process, SIGNAL("readyReadStandardOutput()"), self.OnProcessOutputReady)
#         sublist = self.alist[0]
#         del self.alist[0]
#         print(sublist)
#         self.process.start(sublist[0], sublist[1])
#
#     def OnProcessOutputReady(self):
#         codec = QTextCodec.codecForName("UTF-8")
#         self.txt.setText(self.txt.toPlainText() + codec.toUnicode(self.process.readAllStandardOutput().data()))
#
#     def finished(self, rv):
#         if (len(self.alist) == 0):
#             self.process = None
#             return
#         sublist = self.alist[0]
#         del self.alist[0]
#         print(sublist)
#         self.process.start(sublist[0], sublist[1])
#
# app = QApplication(sys.argv)
# window = MainWindow()
# window.show()
# sys.exit(app.exec_())



if __name__ == '__main__':
    print(math.factorial(1000))