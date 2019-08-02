import sys

__author__ = 'filip@naiser.cz'

from PyQt4 import QtGui, QtCore
from collision_view import CollisionView
import cv2
import os
from gui.img_controls import gui_utils
import pickle
import settings_dialog
from PIL import ImageQt
from utils import visualization_utils

from gui.img_controls import markers
import default_settings

settings = QtCore.QSettings("FERDA")

class CollisionEditor(QtGui.QMainWindow):
    """A tool for results of collision including MSER region editing and
        ant contour fitting."""

    def __init__(self):
        super(CollisionEditor, self).__init__()

        self.drawing_manager = DrawingManager()
        self.pix_map_bg = None
        self.pix_map_region = None
        self.pix_map_avg_ant = None
        self.image = self.load_image('/home/flipajs/dump/collision_editor/frames/209.png')
        self.cv_image = None
        self.pos_marker = None
        self.copy_marker = None
        self.action_list = []

        self.pix_map_offset = 15
        self.region_color = QtGui.QColor(0, 255, 255, 90).rgba()
        self.from_zero = False
        self.region = self.load_region_data()

        self.avg_ant = self.load_avg_ant_data()
        self.ant_number = 2
        self.identity_markers = []
        self.region_temp_pts = list(self.get_pts_from_region(self.region, from_zero=self.from_zero, offset=self.pix_map_offset))

        self.resize(800, 600)
        self.setWindowTitle("Collision Editor")

        self.setCentralWidget(QtGui.QWidget(self))
        self.central_layout = QtGui.QVBoxLayout()
        self.central_layout.setMargin(0)
        self.centralWidget().setLayout(self.central_layout)

        self.setStatusBar(QtGui.QStatusBar())
        self.statusBar().setSizeGripEnabled(False)

        self.setMenuBar(QtGui.QMenuBar())

        self.draw_region_action = QtGui.QAction("Draw region", self.centralWidget())
        self.settings_action = QtGui.QAction("Settings", self.centralWidget())
        self.menuBar().addAction(self.draw_region_action)
        self.menuBar().addAction(self.settings_action)

        self.cancel_action = QtGui.QAction("Cancel fixing", self.centralWidget())
        self.addAction(self.cancel_action)

        self.init_actions()

        self.scene = QtGui.QGraphicsScene()
        self.graphics_view = CollisionView(self.centralWidget())
        self.central_layout.addWidget(self.graphics_view)
        self.graphics_view.setScene(self.scene)

        self.connect_gui()
        self.set_shortcuts()

        self.graphics_view.areaSelected.connect(self.drag_ended)

        self.update()
        self.show()

        self.update_background()


        reg = EditablePixmap(
            self.get_pts_from_region(self.region),
            self.scene,
            self.image.shape[1],
            self.image.shape[0],
            self.region_color
        )
        self.drawing_manager.add_layer('region', reg)

        reg = EditablePixmap(
            self.get_pts_from_region(self.load_region_data(9)),
            self.scene,
            self.image.shape[1],
            self.image.shape[0],
            QtGui.QColor(255, 0, 255, 90).rgba()
        )
        self.drawing_manager.add_layer('ant', reg)
        self.drawing_manager.activate('ant')

        self.drawing_manager.get_layer('region').translate(-40, -100)
        self.drawing_manager.get_layer('region').rotate(90)

        r = self.region
        self.graphics_view.zoom_into(r['roi_tl'][0], r['roi_tl'][1], r['roi_br'][0], r['roi_br'][1])


        self.add_markers()

    def marker_changed(self, id):
        print self.identity_markers[id][0].pos().x()

    def update_avg_ant(self):
        if self.pix_map_avg_ant is not None:
            self.scene.removeItem(self.pix_map_avg_ant)
            self.pix_map_avg_ant = None

        pix_map = self.get_pixmap_from_pts(self.region_temp_pts)

        self.pix_map_avg_ant = self.scene.addPixmap(pix_map)

    def add_markers(self):
        positions = [[[789, 416], [775, 425], [795, 405]], [[789, 416], [775, 425], [795, 405]]]

        for i in range(self.ant_number):
            ant_markers = []

            c = visualization_utils.get_q_color(i, 2)

            item = markers.CenterMarker(0, 0, settings.value('center_marker_size', default_settings.get_default('center_marker_size'), int), c, i, self.marker_changed)
            item.setZValue(0.5)
            ant_markers.append(item)
            self.scene.addItem(item)

            item = markers.HeadMarker(0, 0, settings.value('head_marker_size', default_settings.get_default('head_marker_size'), int), c, i, self.marker_changed)
            item.setZValue(0.5)
            ant_markers.append(item)
            self.scene.addItem(item)

            item = markers.TailMarker(0, 0, settings.value('tail_marker_size', default_settings.get_default('tail_marker_size'), int), c, i, self.marker_changed)
            item.setZValue(0.5)
            ant_markers.append(item)
            self.scene.addItem(item)

            # Connect markers.
            ant_markers[0].add_head_marker(ant_markers[1])
            ant_markers[0].add_tail_marker(ant_markers[2])
            ant_markers[1].add_center_marker(ant_markers[0])
            ant_markers[1].add_other_marker(ant_markers[2])
            ant_markers[2].add_center_marker(ant_markers[0])
            ant_markers[2].add_other_marker(ant_markers[1])
            for j in range(3):
                ant_markers[j].setPos(positions[i][j][0], positions[i][j][1])

            #Remember markers.
            self.identity_markers.append(ant_markers)

    def load_avg_ant_data(self):
        f = open('/home/flipajs/dump/collision_editor/regions/209.pkl', "rb")
        regions = pickle.load(f)

        f.close()

        id_ = 9
        return regions[id_]

    def load_region_data(self, id_ = 17):
        f = open('/home/flipajs/dump/collision_editor/regions/209.pkl', "rb")
        regions = pickle.load(f)

        f.close()

        return regions[id_]

    def get_pts_from_region(self, region, from_zero=False, offset=0):
        pts = []
        for r in region['rle']:
            for c in range(r['col1'], r['col2']+1):
                x = c
                y = r['line']
                if from_zero:
                    x -= region['roi_tl'][0] - offset
                    y -= region['roi_tl'][1] - offset

                pts.append([x, y])

        return pts

    def connect_gui(self):
        """Connects GUI elements with appropriate methods"""
        self.draw_region_action.triggered.connect(self.draw_region)
        self.settings_action.triggered.connect(self.show_settings)

    def set_shortcuts(self):
        """Sets shortcuts for actions"""
        self.draw_region_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_D))

    def init_actions(self):
        self.action_list.append(self.draw_region_action)
        self.action_list.append(self.settings_action)
        self.draw_region_action.setObjectName('draw_region')
        self.settings_action.setObjectName('settings')

    def draw_region(self):
        self.graphics_view.set_drawing_mode(self.drawing_manager.drawing_signal)

    def load_image(self, filename='test2.png'):
        filename = os.path.normpath(filename)

        np_img = cv2.imread(filename)
        return np_img

    def show_settings(self):
        dialog = settings_dialog.SettingsDialog(self, actionlist=self.action_list)
        dialog.exec_()

    def drag_ended(self, point_one, point_two):
        if self.copy_marker is None:
            if self.pos_marker is not None:
                self.remove_pos_marker()
            self.add_pos_marker(point_one, point_two)

    def add_pos_marker(self, point_one, point_two):
        settings = QtCore.QSettings("Background corrector")

        point_one = self.graphics_view.mapToScene(point_one)
        point_two = self.graphics_view.mapToScene(point_two)
        width = point_two.x() - point_one.x()
        height = point_two.y() - point_one.y()
        pen = QtGui.QPen()
        pen.setStyle(QtCore.Qt.DashDotLine)
        pen.setWidth(settings.value('square_line_width', settings_dialog.get_default('square_line_width'), int))
        pen.setColor(settings.value('position_square_color', settings_dialog.get_default('position_square_color'), QtGui.QColor))
        brush = QtGui.QBrush()
        brush.setStyle(QtCore.Qt.NoBrush)
        self.pos_marker = QtGui.QGraphicsRectItem(0, 0, width, height)
        self.pos_marker.setPos(point_one)
        self.pos_marker.setBrush(brush)
        self.pos_marker.setPen(pen)
        self.pos_marker.setZValue(.5)
        self.scene.addItem(self.pos_marker)

    def add_copy_marker(self, x, y, width, height):
        settings = QtCore.QSettings("Background corrector")

        pen = QtGui.QPen()
        pen.setStyle(QtCore.Qt.DashDotLine)
        pen.setWidth(settings.value('square_line_width', settings_dialog.get_default('square_line_width'), int))
        pen.setColor(settings.value('copy_square_color', settings_dialog.get_default('copy_square_color'), QtGui.QColor))
        brush = QtGui.QBrush()
        brush.setStyle(QtCore.Qt.NoBrush)
        self.copy_marker = QtGui.QGraphicsRectItem(0, 0, width, height)
        self.copy_marker.setPos(x, y)
        self.copy_marker.setBrush(brush)
        self.copy_marker.setPen(pen)
        self.copy_marker.setZValue(.6)
        self.copy_marker.setFlag(QtGui.QGraphicsItem.ItemIsMovable, True)
        self.scene.addItem(self.copy_marker)

    def remove_pos_marker(self):
        self.scene.removeItem(self.pos_marker)
        self.pos_marker = None

    def remove_copy_marker(self):
        self.scene.removeItem(self.copy_marker)
        self.copy_marker = None

    def cancel_fixing(self):
        if self.copy_marker is not None:
            self.remove_copy_marker()

    def correct_background(self):
        settings = QtCore.QSettings("Background corrector")

        blur_distance = settings.value('blur_distance', settings_dialog.get_default('blur_distance'), int)

        img_height, img_width, img_depth = self.image.shape
        width = int(self.copy_marker.rect().width())
        height = int(self.copy_marker.rect().height())
        new_x = int(self.pos_marker.scenePos().x())
        new_y = int(self.pos_marker.scenePos().y())
        orig_x = int(self.copy_marker.scenePos().x())
        orig_y = int(self.copy_marker.scenePos().y())

        if new_x < 0 or new_y < 0 or orig_x < 0 or orig_y < 0 or new_x + width > img_width or new_y + height > img_height or orig_x + width > img_width or orig_y + height > img_height:
            return False

        #copy image
        part = self.get_img_part(orig_x, orig_y, width, height)
        self.set_img_part(new_x, new_y, width, height, part)

        #blur edges
        x_blur = max(0, new_x - blur_distance)
        y_blur = max(0, new_y - blur_distance)
        w_blur = min(img_width, width + 2*blur_distance)
        h_blur = min(img_height, height + 2*blur_distance)
        x_keep = min((2*new_x + width)/2, new_x + blur_distance)
        y_keep = min((2*new_y + height)/2, new_y + blur_distance)
        w_keep = max((2*new_x + width)/2, width - 2*blur_distance)
        h_keep = max((2*new_y + height)/2, height - 2*blur_distance)

        to_blur = self.get_img_part(x_blur, y_blur, w_blur, h_blur)
        to_keep = self.get_img_part(x_keep, y_keep, w_keep, h_keep)

        blurred = cv2.blur(to_blur, (5, 5))

        self.set_img_part(x_blur, y_blur, w_blur, h_blur, blurred)
        self.set_img_part(x_keep, y_keep, w_keep, h_keep, to_keep)

        self.update_background()
        return True

    def update_background(self):
        if self.pix_map_bg is not None:
            self.scene.removeItem(self.pix_map_bg)
            self.pix_map_bg = None
        pix_map = gui_utils.cvimg2qtpixmap(self.image)
        gui_utils.view_add_bg_image(self.graphics_view, pix_map)
        self.pix_map_bg = self.scene.addPixmap(pix_map)

    def get_pixmap_from_pts(self, pts):
        img_q = self.draw_points_onto_layer(pts)
        pix_map = QtGui.QPixmap.fromImage(img_q)

        return pix_map

    def update_region_mask(self):
        if self.pix_map_region is not None:
            self.scene.removeItem(self.pix_map_region)
            self.pix_map_region = None

        pix_map = self.get_pixmap_from_pts(self.region_temp_pts)

        self.pix_map_region = self.scene.addPixmap(pix_map)

    def draw_points_onto_layer(self, pts):
        r = self.region

        im_width = self.image.shape[1]
        im_height = self.image.shape[0]
        if self.from_zero:
            im_width = r['roi_br'][0] - r['roi_tl'][0] + 2 * self.pix_map_offset
            im_height = r['roi_br'][1] - r['roi_tl'][1] + 2 * self.pix_map_offset

        ImageQt.QImage()
        img_q = ImageQt.QImage(im_width, im_height, QtGui.QImage.Format_ARGB32)
        img_q.fill(QtGui.QColor(0, 0, 0, 0).rgba())

        for pt in pts:
            img_q.setPixel(pt[0], pt[1], self.region_color)

        return img_q

    def get_img_part(self, x, y, width, height):
        part = self.image[y: y + height, x: x + width]
        return part

    def set_img_part(self, x, y, width, height, part):
        self.image[y: y + height, x: x + width] = part



def main():
    app = QtGui.QApplication(sys.argv)
    ex = CollisionEditor()

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()