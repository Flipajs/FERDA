__author__ = 'filip@naiser.cz'
from PyQt4 import QtGui
import ImageQt
from numpy import *

import cv2
from utils import video_manager

import visualization_utils


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

    def visualize_new_data(self, frame, id_manager, new_data, width=200, height=200):
        gui = QtGui.QApplication.processEvents

        i = 1
        j = 1

        length = len(new_data)

        img = self.video.seek_frame(frame)
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

            item = QtGui.QLabel()

            item.setScaledContents(True)
            item.setFixedWidth(pix_map.width())
            item.setFixedHeight(pix_map.height())
            item.setPixmap(pix_map)

            self.grid.addWidget(item, j, i)
            i += 1
            if i % 4 == 0:
                i = 1
                j += 1

            if j < 10:
                gui()

            img = self.video.move2_next()

    def update_sequence(self, frame, length, id_manager, ant_id, width=200, height=200):
        # obtaining copy of video solves multiple calling of this method (calling from gui is asynchronous,
        # thus wrong frame number might be read during another call of this method.
        local_vid = self.video.get_manager_copy()

        gui = QtGui.QApplication.processEvents

        changes = self.get_changes(frame, id_manager, ant_id)

        i = 1
        j = 1

        #first frame is not visualized, that is the reason for frame+1
        img = local_vid.seek_frame(frame + 1)
        for f in range(frame + 1, frame + length):
            self.draw_ants(img, f, id_manager)

            pos = id_manager.get_positions(f, ant_id)

            keys = ['hx', 'hy', 'cy', 'cx', 'bx', 'by']
            for k in keys:
                pos[k] -= changes[k]


            self.draw_data(img, pos)


            img_ = zeros((shape(img)[0] + 2 * height, shape(img)[1] + 2 * width, 3), dtype=uint8)
            img_[height:-height, width:-width] = img.copy()

            x = pos['cx'] + width / 2
            y = pos['cy'] + height / 2

            crop = img_[y:y + height, x:x + width, :].copy()

            img_q = ImageQt.QImage(crop.data, crop.shape[1], crop.shape[0], crop.shape[1] * 3, 13)
            pix_map = QtGui.QPixmap.fromImage(img_q.rgbSwapped())

            item = QtGui.QLabel()

            item.setScaledContents(True)
            item.setFixedWidth(pix_map.width())
            item.setFixedHeight(pix_map.height())
            item.setPixmap(pix_map)

            self.grid.addWidget(item, j, i)
            i += 1
            if i % 4 == 0:
                i = 1
                j += 1

            if j < 10:
                gui()

            img = local_vid.move2_next()


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