from PyQt6 import QtCore, QtWidgets
from PyQt6 import QtGui, QtWidgets


class ControlPanel(QtWidgets.QFrame):

    def __init__(self, graph_w_callback):
        super(ControlPanel, self).__init__()
        # self.setStyleSheet("background-color: rgb(255,255,255); margin:5px; border:1px solid rgb(0, 0, 0); ")
        self.l = QtWidgets.QHBoxLayout()
        self.l.setContentsMargins(20, 0, 20, 0)
        self.prepare_buttons(graph_w_callback)
        self.setLayout(self.l)

    def prepare_button(self, label, callback, key):
        button = QtWidgets.QPushButton(label)
        action = QtGui.QAction(self)
        action.triggered.connect(callback)
        button.clicked.connect(callback)
        action.setShortcut(QtGui.QKeySequence(key))
        self.addAction(action)
        self.l.addWidget(button)

    def prepare_buttons(self, graph_w_callback):
        self.prepare_button('Show info (A)', graph_w_callback.info_manager.show_all_info, QtCore.Qt.Key.Key_A)
        self.prepare_button('Hide info (S)', graph_w_callback.info_manager.hide_all_info, QtCore.Qt.Key.Key_S)
        self.prepare_button('Show zoomed (T)', graph_w_callback.node_zoom_manager.show_zoom_all, QtCore.Qt.Key.Key_T)
        self.prepare_button('Hide zoomed (Y)', graph_w_callback.node_zoom_manager.hide_zoom_all, QtCore.Qt.Key.Key_Y)
        self.prepare_button('Zoom to chunk id (L)', graph_w_callback.zoom_to_chunk_event, QtCore.Qt.Key.Key_L)
        self.prepare_button('Change Display (F5)', graph_w_callback.toggle_show_vertically, QtCore.Qt.Key.Key_F5)
        self.prepare_button('Decompress Display (F4)', graph_w_callback.compress_axis_toggle, QtCore.Qt.Key.Key_F4)
        self.prepare_button('Stretch (N)', graph_w_callback.stretch, QtCore.Qt.Key.Key_N)
        self.prepare_button('Shrink (M)', graph_w_callback.shrink, QtCore.Qt.Key.Key_M)
        self.prepare_button('Update colors (C)', graph_w_callback.update_lines, QtCore.Qt.Key.Key_C)
        self.prepare_button('show images (I)', graph_w_callback.show_node_images, QtCore.Qt.Key.Key_I)
