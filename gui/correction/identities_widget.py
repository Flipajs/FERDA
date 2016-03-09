from PyQt4 import QtGui, QtCore
from gui.img_controls.my_view import MyView
from utils.video_manager import get_auto_video_manager
from gui.img_controls.utils import cvimg2qtpixmap
import math
import cv2
from viewer.gui.img_controls import markers
from core.animal import colors_
from core.settings import Settings as S_
from core.graph.region_chunk import RegionChunk
import numpy as np
import sys
from core.animal import Animal
from gui import gui_utils

class AnimalVisu(QtGui.QWidget):
    def __init__(self, animal):
        super(AnimalVisu, self).__init__()

        self.hbox = QtGui.QHBoxLayout()
        self.setLayout(self.hbox)

        cimg = np.zeros((15, 10, 3), dtype=np.uint8)
        cimg = np.asarray(cimg+animal.color_, dtype=np.uint8)
        self.color_label = gui_utils.get_image_label(cimg)

        self.hbox.addWidget(self.color_label)
        self.hbox.addWidget(QtGui.QLabel(animal.name))

        self.orig_img = None
        self.adjusted_img = None

    def update_visu(self, img, region, project):
        if region is None:
            # set gray images
            return

        from utils.img import rotate_img, centered_crop, get_bounding_box, endpoint_rot
        relative_border = 5.1
        border = 50

        bb, offset = get_bounding_box(region, project, relative_border, img)

        from utils.img import get_safe_selection
        roi_ = region.roi()
        x = roi_.x() - border
        y = roi_.y() - border
        h_ = roi_.height() + 2 * border
        w_ = roi_.width() + 2 * border

        bb = get_safe_selection(img, y, x, h_, w_)

        self.color_label1 = gui_utils.get_image_label(bb)
        self.hbox.addWidget(self.color_label1)
        bb = rotate_img(bb, region.theta_)
        bb = centered_crop(bb, 8*region.b_, 4*region.a_)

        import matplotlib as mpl
        bb_hsv = mpl.colors.rgb_to_hsv(bb)
        bb_hsv[:,:,1] *= 2.0
        bb_hsv[:,:,2] *= 1.2
        bb = mpl.colors.hsv_to_rgb(bb_hsv)

        bb = np.asarray(np.clip(bb - bb.min(), 0, 255), dtype=np.uint8)

        self.color_label = gui_utils.get_image_label(bb)
        self.hbox.addWidget(self.color_label)



class IdentitiesWidget(QtGui.QWidget):
    def __init__(self, project):
        super(IdentitiesWidget, self).__init__()

        self.p = project

        # TOOD: remove in future
        self.p.animals = [
            # BGR colors
            Animal(0, 'light blue', color=(255, 100, 70)),
            Animal(1, 'red', color=(0, 0, 255)),
            Animal(2, 'yellow', color=(0, 255, 255)),
            Animal(3, 'green', color=(0, 255, 0)),
            Animal(4, 'dark blue', color=(230, 0, 0)),
            Animal(5, 'silver', color=(230, 230, 230))
        ]

        self.vbox = QtGui.QVBoxLayout()
        self.setLayout(self.vbox)

        self.animal_widgets = []
        for a in self.p.animals:
            w_ = AnimalVisu(a)
            self.animal_widgets.append(w_)
            self.vbox.addWidget(w_)

    def update(self, frame):
        regions = [None] * len(self.p.animals)

        from utils.video_manager import get_auto_video_manager
        from core.graph.region_chunk import RegionChunk
        vm = get_auto_video_manager(self.p)
        chunks = self.p.gm.chunks_in_frame(frame)
        img = vm.get_frame(frame)

        for ch in chunks:
            r_ch = RegionChunk(ch, self.p.gm, self.p.rm)
            regions[ch.animal_id_] = r_ch.region_in_t(frame)

        for i in range(len(self.p.animals)):
            self.animal_widgets[i].update_visu(img, regions[i], self.p)


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    from core.project.project import Project
    import cPickle as pickle

    project = Project()
    name = 'Cam1_orig'
    wd = '/Users/flipajs/Documents/wd/GT/'
    project.load(wd+name+'/cam1.fproj')

    with open(project.working_directory+'/temp/animal_id_mapping.pkl', 'rb') as f_:
        animal_id_mapping = pickle.load(f_)

    for ch_id in project.gm.chunk_list():
        animal_id = -1
        if ch_id in animal_id_mapping:
            animal_id = animal_id_mapping[ch_id]

        project.chm[ch_id].animal_id_ = animal_id

    ex = IdentitiesWidget(project)
    ex.update(3000)
    ex.show()

    app.exec_()
    app.deleteLater()
    sys.exit()

