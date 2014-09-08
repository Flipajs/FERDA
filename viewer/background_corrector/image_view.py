from gui.img_controls.my_view import MyView
from PyQt4.QtCore import *


class ImageView(MyView):

	areaSelected = pyqtSignal("PyQt_PyObject", "PyQt_PyObject")

	def __init__(self, parent=None):
		super(ImageView, self).__init__(parent)
		# self.setDragMode(self.NoDrag)
		self.point_one = None
		self.point_two = None

	def mousePressEvent(self,  event):
		super(ImageView, self).mousePressEvent(event)
		if event.buttons() == Qt.LeftButton:
			self.point_one = event.pos()

	def mouseReleaseEvent(self, event):
		super(ImageView, self).mouseReleaseEvent(event)
		if self.point_one is not None:
			self.point_two = event.pos()
			if self.point_one.x() > self.point_two.x():
				tmp = self.point_one.x()
				self.point_one.setX(self.point_two.x())
				self.point_two.setX(tmp)
			if self.point_one.y() > self.point_two.y():
				tmp = self.point_one.y()
				self.point_one.setY(self.point_two.y())
				self.point_two.setY(tmp)
			self.areaSelected.emit(self.point_one, self.point_two)