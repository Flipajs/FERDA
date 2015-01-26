__author__ = 'filip@naiser.cz'
from PyQt4 import QtGui, QtCore
import ImageQt
from numpy import *

import cv2
from utils import video_manager

import visualization_utils


class SelectableImageQLabel(QtGui.QLabel):
    def __init__(self, parent=None, selected_callback=None, frame=-1):
        QtGui.QLabel.__init__(self, parent)
        self.frame_number = frame
        self.selected_callback = selected_callback

    def mouseReleaseEvent(self, ev):
        self.setStyleSheet("border: 2px dashed black;")
        self.selected_callback(self, self.frame_number)

    def deselect(self):
        self.setStyleSheet("border: 0px;")


class ImgSequenceWidget(QtGui.QWidget):
    def __init__(self, video_manager):
        super(ImgSequenceWidget, self).__init__()
        self.video = video_manager
        self.crop_width = 100
        self.crop_height = 100

        self.main_layout = QtGui.QVBoxLayout()
        self.setLayout(self.main_layout)

        self.scroll_area = QtGui.QScrollArea()
        self.scroll_area.setWidgetResizable(True)

        self.scroll_area_content = QtGui.QWidget()

        self.scroll_area_content_vlayout = QtGui.QVBoxLayout(self.scroll_area_content)
        self.scroll_area_content_vlayout.setSpacing(0)
        self.scroll_area_content_vlayout.setMargin(0)

        self.grid = QtGui.QGridLayout()
        self.grid.setSizeConstraint(QtGui.QLayout.SetNoConstraint)
        self.grid.setSpacing(1)

        self.scroll_area_content_vlayout.addLayout(self.grid)
        self.scroll_area.setWidget(self.scroll_area_content)

        self.main_layout.addWidget(self.scroll_area)

        # apply button
        self.apply_new_positions_button = QtGui.QPushButton()
        self.apply_new_positions_button.setText('apply changes selected included')
        self.apply_new_positions_button.clicked.connect(self.apply_new_positions_clicked)

        self.main_layout.addWidget(self.apply_new_positions_button)

        self.selected = None
        self.selected_frame = -1

        self.scroll_area.verticalScrollBar().valueChanged.connect(self.scroll_changed)


        # store these values so it is possible to add new data when asked for.
        self.grid_i = -1
        self.grid_j = -1
        self.frame = None
        self.local_vid = None
        self.id_manager = None
        self.ant_id = -1
        self.im_height_ = -1
        self.im_width_ = -1
        self.changes = None
        self.new_data = None

        self.update()
        self.show()


    def scroll_changed(self):
        s = self.scroll_area.verticalScrollBar()
        val = s.value() / float(s.maximum())
        print "Scrolling", val

        length = 30
        if val > 0.8:
            #TODO: add scroll_callback
            # prepare grid with blank images (so there is no blinking during rewriting or adding more rows...

            gui = QtGui.QApplication.processEvents

            img = self.local_vid.seek_frame(self.frame + 1)
            for f in range(self.frame + 1, self.frame + length):
                self.draw_ants(img, f, self.id_manager)

                pos = self.id_manager.get_positions(f, self.ant_id)

                keys = ['hx', 'hy', 'cy', 'cx', 'bx', 'by']
                for k in keys:
                    pos[k] -= self.new_data[k]


                self.draw_data(img, pos)


                img_ = zeros((shape(img)[0] + 2 * self.im_height_, shape(img)[1] + 2 * self.im_width_, 3), dtype=uint8)
                img_[self.im_height_:-self.im_height_, self.im_width_:-self.im_width_] = img.copy()

                x = pos['cx'] + self.im_width_ / 2
                y = pos['cy'] + self.im_height_ / 2

                crop = img_[y:y + self.im_height_, x:x + self.im_width_, :].copy()

                img_q = ImageQt.QImage(crop.data, crop.shape[1], crop.shape[0], crop.shape[1] * 3, 13)
                pix_map = QtGui.QPixmap.fromImage(img_q.rgbSwapped())

                item = QtGui.QLabel()

                item.setScaledContents(True)
                item.setFixedself.im_width_(pix_map.self.im_width_())
                item.setFixedself.im_height_(pix_map.self.im_height_())
                item.setPixmap(pix_map)

                self.grid.addWidget(item, self.grid_j, self.grid_i)
                self.grid_i += 1
                if self.grid_i % 4 == 0:
                    self.grid_i = 1
                    self.grid_j += 1

                if self.grid_j < 10:
                    gui()

                img = self.local_vid.move2_next()

            gui()


    def selected_callback(self, selected, frame):
        if self.selected:
            self.selected.deselect()

        self.selected = selected
        self.selected_frame = frame

    def get_changes(self, frame, id_manager, ant_id):
        changes = {}
        keys = ['hx', 'hy', 'cy', 'cx', 'bx', 'by']

        for k in keys:
            changes[k] = 0

        if frame in id_manager.changes_for_frames:
            for c in id_manager.changes_for_frames[frame]:
                if ant_id in c['change_data']:
                    chan = c['change_data'][ant_id]
                    orig = c['old_data'][ant_id]

                    for k in keys:
                        if orig[k] and chan[k]:
                            changes[k] += orig[k] - chan[k]

        return changes

    def draw_ants(self, img, frame, id_manager):
        for a_id in range(id_manager.ant_num):
            r, g, b = visualization_utils.get_color(a_id, id_manager.ant_num)
            pos = id_manager.get_positions(frame, a_id)

            self.draw_data(img, pos, r, g, b)

    def draw_data(self, img, data, r=255, g=255, b=255):
        keys = ['hx', 'hy', 'cx', 'cy', 'bx', 'by']

        for i in range(3):
            cv2.circle(img, (int(data[keys[2 * i]]), int(data[keys[2 * i + 1]])), 3, (b, g, r), -1)

    def add_frame(self, frame, ant_id, id_manager, new_data, width=200, heigth=200):
        return

    def visualize_new_data(self, frame, ant_id, id_manager, new_data, width=200, height=200):
        gui = QtGui.QApplication.processEvents

        #store this for case of apply changes button click
        self.new_data = new_data
        self.ant_id = ant_id

        self.selected = None
        self.local_vid = self.video.get_manager_copy()
        self.frame = frame
        self.id_manager = id_manager
        self.im_width_ = width
        self.im_height_ = height

        self.grid_i = 1
        self.grid_j = 1

        length = len(new_data)

        img = self.local_vid.seek_frame(frame)
        for f in range(frame, frame + length):
            self.draw_ants(img, f, id_manager)
            self.draw_data(img, new_data[f-frame])

            img_ = zeros((shape(img)[0] + 2 * height, shape(img)[1] + 2 * width, 3), dtype=uint8)
            img_[height:-height, width:-width] = img.copy()

            x = new_data[f - frame]['cx'] + width / 2
            y = new_data[f - frame]['cy'] + height / 2

            crop = img_[y:y + height, x:x + width, :].copy()

            img_q = ImageQt.QImage(crop.data, crop.shape[1], crop.shape[0], crop.shape[1] * 3, 13)
            pix_map = QtGui.QPixmap.fromImage(img_q.rgbSwapped())

            item = SelectableImageQLabel(self, self.selected_callback, f)

            item.setScaledContents(True)
            item.setFixedWidth(pix_map.width())
            item.setFixedHeight(pix_map.height())
            item.setPixmap(pix_map)

            self.grid.addWidget(item, self.grid_j, self.grid_i)
            self.grid_i += 1
            if self.grid_i % 4 == 0:
                self.grid_i = 1
                self.grid_j += 1

            if self.grid_j < 20:
                gui()

            img = self.local_vid.move2_next()


    def update_sequence(self, frame, length, id_manager, ant_id, width=200, height=200):
        #storing these values so they can be used when it is asked for more self.frames

        # obtaining copy of video solves multiple calling of this method (calling from gui is asynchronous,
        # thus wrong self.frame number might be read during another call of this method.
        self.local_vid = self.video.get_manager_copy()
        self.frame = frame
        self.id_manager = id_manager
        self.ant_id = ant_id
        self.im_width_ = width
        self.im_height_ = height
        

        gui = QtGui.QApplication.processEvents

        self.changes = self.get_changes(self.frame, self.id_manager, self.ant_id)

        self.grid_i = 1
        self.grid_j = 1

        #first self.frame is not visualized, that is the reason for self.frame+1
        img = self.local_vid.seek_frame(self.frame + 1)
        for f in range(self.frame + 1, self.frame + length):
            self.draw_ants(img, f, self.id_manager)

            pos = self.id_manager.get_positions(f, self.ant_id)

            keys = ['hx', 'hy', 'cy', 'cx', 'bx', 'by']
            for k in keys:
                pos[k] -= self.changes[k]


            self.draw_data(img, pos)


            img_ = zeros((shape(img)[0] + 2 * self.im_height_, shape(img)[1] + 2 * self.im_width_, 3), dtype=uint8)
            img_[self.im_height_:-self.im_height_, self.im_width_:-self.im_width_] = img.copy()

            x = pos['cx'] + self.im_width_ / 2
            y = pos['cy'] + self.im_height_ / 2

            crop = img_[y:y + self.im_height_, x:x + self.im_width_, :].copy()

            img_q = ImageQt.QImage(crop.data, crop.shape[1], crop.shape[0], crop.shape[1] * 3, 13)
            pix_map = QtGui.QPixmap.fromImage(img_q.rgbSwapped())

            item = QtGui.QLabel()

            item.setScaledContents(True)
            item.setFixedself.im_width_(pix_map.self.im_width_())
            item.setFixedself.im_height_(pix_map.self.im_height_())
            item.setPixmap(pix_map)

            self.grid.addWidget(item, self.grid_j, self.grid_i)
            self.grid_i += 1
            if self.grid_i % 4 == 0:
                self.grid_i = 1
                self.grid_j += 1

            if self.grid_j < 10:
                gui()

            img = self.local_vid.move2_next()

        self.frame = self.frame + length

    def apply_new_positions_clicked(self):
        if not self.selected:
            return

        for i in range(self.frame, self.selected_frame+1):
            p = self.id_manager.get_positions(i, self.ant_id)

            for k in self.new_data[i-self.frame]:
                p[k] = self.new_data[i-self.frame][k]

        print self.selected_frame

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