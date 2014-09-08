from PyQt4 import Qt

__author__ = 'flipajs'

#from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import *
import PyQt4.Qt
from PyQt4.QtCore import *


class MyView(QGraphicsView, object):
	def __init__(self, parent=None):
		super(MyView, self).__init__(parent)
		self.setMouseTracking(True)
		self.scale(1, 1)
		self.startPos = None
		self.mouse_pressed = False
		self.start_position = (0, 0)
		self.setDragMode(QGraphicsView.RubberBandDrag)
		self._isPanning = False
		self._mousePressed = False

	def mousePressEvent(self,  event):
		if event.button() == Qt.RightButton:
			self.setCursor(Qt.ClosedHandCursor)
			self._dragPos = event.pos()
			self._isPanning = True
			event.accept()
		else:
			super(MyView, self).mousePressEvent(event)

	def mouseMoveEvent(self, event):
		if self._isPanning:
			newPos = event.pos()
			diff = newPos - self._dragPos
			self._dragPos = newPos
			self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - diff.x())
			self.verticalScrollBar().setValue(self.verticalScrollBar().value() - diff.y())
			event.accept()
		else:
			super(MyView, self).mouseMoveEvent(event)

	def mouseReleaseEvent(self, event):
		if event.button() == Qt.RightButton:
			self.setCursor(Qt.ArrowCursor)
			self._isPanning = False
		super(MyView, self).mouseReleaseEvent(event)

	def mouseDoubleClickEvent(self, event):
		pass

	# def keyPressEvent(self, event):
	# 	if event.key() == Qt.Key_Control and not self._mousePressed:
	# 		self._isPanning = True
	# 		self.setCursor(Qt.OpenHandCursor)
	# 	else:
	# 		super(MyView, self).keyPressEvent(event)
	#
	# def keyReleaseEvent(self, event):
	# 	if event.key() == Qt.Key_Control:
	# 		if not self._mousePressed:
	# 			self._isPanning = False
	# 			self.setCursor(Qt.ArrowCursor)
	# 	else:
	# 		super(MyView, self).keyPressEvent(event)


	def wheelEvent(self, event):
		scale_factor = 1.15
		numSteps = event.delta() / 15 / 8

		if numSteps == 0:
			event.ignore()
			return

		sc = scale_factor**numSteps

		m11 = self.transform().m11()
		m22 = self.transform().m22()

		if (m11 < 0.1 or m22 < 0.1) and sc < 1:
			return

		self.zoom(sc, self.mapToScene(event.pos()))

	def zoom(self, factor, center_point):
		self.scale(factor, factor)
		self.centerOn(center_point)