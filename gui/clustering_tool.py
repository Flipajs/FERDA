from PyQt4 import QtGui, QtCore
import sys
from core.region.clustering import clustering, display_cluster_representants, draw_region
import cPickle as pickle
from sklearn.preprocessing import StandardScaler
from utils.video_manager import get_auto_video_manager
from utils.drawing.collage import create_collage_rows
from scipy.spatial.distance import cdist
import numpy as np
import cv2
from gui.img_grid.img_grid_widget import ImgGridWidget
from functools import partial
from utils.misc import print_progress


# TODO: save labels + versioning
# TODO: file for images, file for labels
# TODO: select all until

class ClusteringTool(QtGui.QWidget):
    def __init__(self, p):
        super(ClusteringTool, self).__init__()

        self.p = p

        self.WW = 150
        self.HH = 150
        self.COLS = 3


        self.vbox = QtGui.QVBoxLayout()
        self.setLayout(self.vbox)

        self.b = QtGui.QPushButton('test')
        self.vbox.addWidget(self.b)

        self.singles = ImgGridWidget(cols=1, element_width=self.WW)
        self.multi = ImgGridWidget(cols=1, element_width=self.WW)
        self.noise = ImgGridWidget(cols=1, element_width=self.WW)
        self.part = ImgGridWidget(cols=1, element_width=self.WW)

        self.hbox = QtGui.QHBoxLayout()
        self.vbox.addLayout(self.hbox)

        self.hbox_buttons = QtGui.QHBoxLayout()
        self.vbox.addLayout(self.hbox_buttons)

        self.to_single_b = QtGui.QPushButton('to singles')
        self.to_single_b.clicked.connect(partial(self.move_selected_to, 'single'))
        self.hbox_buttons.addWidget(self.to_single_b)

        self.to_single_a = QtGui.QAction('to single', self)
        self.to_single_a.triggered.connect(partial(self.move_selected_to, 'single'))
        self.to_single_a.setShortcut(QtGui.QKeySequence(QtCore.Qt.SHIFT + QtCore.Qt.Key_S))
        self.addAction(self.to_single_a)
        
        self.to_multi_b = QtGui.QPushButton('to multis')
        self.to_multi_b.clicked.connect(partial(self.move_selected_to, 'multi'))
        self.hbox_buttons.addWidget(self.to_multi_b)

        self.to_multi_a = QtGui.QAction('to multi', self)
        self.to_multi_a.triggered.connect(partial(self.move_selected_to, 'multi'))
        self.to_multi_a.setShortcut(QtGui.QKeySequence(QtCore.Qt.SHIFT + QtCore.Qt.Key_M))
        self.addAction(self.to_multi_a)

        self.to_noise_b = QtGui.QPushButton('to noises')
        self.to_noise_b.clicked.connect(partial(self.move_selected_to, 'noise'))
        self.hbox_buttons.addWidget(self.to_noise_b)

        self.to_noise_a = QtGui.QAction('to noise', self)
        self.to_noise_a.triggered.connect(partial(self.move_selected_to, 'noise'))
        self.to_noise_a.setShortcut(QtGui.QKeySequence(QtCore.Qt.SHIFT + QtCore.Qt.Key_N))
        self.addAction(self.to_noise_a)

        self.to_part_b = QtGui.QPushButton('to parts')
        self.to_part_b.clicked.connect(partial(self.move_selected_to, 'part'))
        self.hbox_buttons.addWidget(self.to_part_b)
        
        self.save_b = QtGui.QPushButton('save')
        self.save_b.clicked.connect(self.save)
        self.hbox_buttons.addWidget(self.save_b)

        # SELECT ALL ACTIONS
        self.to_part_a = QtGui.QAction('to part', self)
        self.to_part_a.triggered.connect(partial(self.move_selected_to, 'part'))
        self.to_part_a.setShortcut(QtGui.QKeySequence(QtCore.Qt.SHIFT + QtCore.Qt.Key_P))
        self.addAction(self.to_part_a)

        self.select_all_singles_a = QtGui.QAction('select all', self)
        self.select_all_singles_a.triggered.connect(partial(self.select_all, 'single'))
        self.select_all_singles_a.setShortcut(QtGui.QKeySequence(QtCore.Qt.SHIFT + QtCore.Qt.CTRL + QtCore.Qt.Key_S))
        self.addAction(self.select_all_singles_a)

        self.select_all_multi_a = QtGui.QAction('select all', self)
        self.select_all_multi_a.triggered.connect(partial(self.select_all, 'multi'))
        self.select_all_multi_a.setShortcut(QtGui.QKeySequence(QtCore.Qt.SHIFT + QtCore.Qt.CTRL + QtCore.Qt.Key_M))
        self.addAction(self.select_all_multi_a)

        self.select_all_noise_a = QtGui.QAction('select all', self)
        self.select_all_noise_a.triggered.connect(partial(self.select_all, 'noise'))
        self.select_all_noise_a.setShortcut(QtGui.QKeySequence(QtCore.Qt.SHIFT + QtCore.Qt.CTRL + QtCore.Qt.Key_N))
        self.addAction(self.select_all_noise_a)

        self.select_until_part_a = QtGui.QAction('select until', self)
        self.select_until_part_a.triggered.connect(partial(self.select_until, 'part'))
        self.select_until_part_a.setShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL + QtCore.Qt.Key_P))
        self.addAction(self.select_until_part_a)

        self.select_until_singles_a = QtGui.QAction('select until', self)
        self.select_until_singles_a.triggered.connect(partial(self.select_until, 'single'))
        self.select_until_singles_a.setShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL + QtCore.Qt.Key_S))
        self.addAction(self.select_until_singles_a)

        self.select_until_multi_a = QtGui.QAction('select until', self)
        self.select_until_multi_a.triggered.connect(partial(self.select_until, 'multi'))
        self.select_until_multi_a.setShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL + QtCore.Qt.Key_M))
        self.addAction(self.select_until_multi_a)

        self.select_until_noise_a = QtGui.QAction('select until', self)
        self.select_until_noise_a.triggered.connect(partial(self.select_until, 'noise'))
        self.select_until_noise_a.setShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL + QtCore.Qt.Key_N))
        self.addAction(self.select_until_noise_a)

        self.select_until_part_a = QtGui.QAction('select until', self)
        self.select_until_part_a.triggered.connect(partial(self.select_until, 'part'))
        self.select_until_part_a.setShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL + QtCore.Qt.Key_P))
        self.addAction(self.select_until_part_a)

        # CHECKBOXES
        self.show_decided = QtGui.QCheckBox('show decided')
        self.show_decided.setChecked(False)
        self.show_decided.stateChanged.connect(self.redraw_grids)
        self.hbox_buttons.addWidget(self.show_decided)

        self.show_undecided = QtGui.QCheckBox('show undecided')
        self.show_undecided.setChecked(True)
        self.show_undecided.stateChanged.connect(self.redraw_grids)
        self.hbox_buttons.addWidget(self.show_undecided)

        self.hbox_check = QtGui.QHBoxLayout()
        self.vbox.addLayout(self.hbox_check)

        self.ch_area = QtGui.QCheckBox('area')
        self.ch_area.setChecked(True)
        self.ch_area.stateChanged.connect(self.redraw_grids)
        self.hbox_check.addWidget(self.ch_area)

        self.ch_a = QtGui.QCheckBox('a')
        self.ch_a.setChecked(True)
        self.ch_a.stateChanged.connect(self.redraw_grids)
        self.hbox_check.addWidget(self.ch_a)

        self.ch_b = QtGui.QCheckBox('b')
        self.ch_b.setChecked(True)
        self.ch_b.stateChanged.connect(self.redraw_grids)
        self.hbox_check.addWidget(self.ch_b)

        self.ch_min_i = QtGui.QCheckBox('min_i')
        self.ch_min_i.setChecked(False)
        self.ch_min_i.stateChanged.connect(self.redraw_grids)
        self.hbox_check.addWidget(self.ch_min_i)

        self.ch_max_i = QtGui.QCheckBox('max_i')
        self.ch_max_i.setChecked(False)
        self.ch_max_i.stateChanged.connect(self.redraw_grids)
        self.hbox_check.addWidget(self.ch_max_i)

        self.ch_margin = QtGui.QCheckBox('margin')
        self.ch_margin.setChecked(False)
        self.ch_margin.stateChanged.connect(self.redraw_grids)
        self.hbox_check.addWidget(self.ch_margin)

        self.ch_c_len = QtGui.QCheckBox('c_len')
        self.ch_c_len.setChecked(True)
        self.ch_c_len.stateChanged.connect(self.redraw_grids)
        self.hbox_check.addWidget(self.ch_c_len)

        self.grids = {'single': self.singles, 'multi': self.multi, 'noise': self.noise, 'part': self.part}

        self.compute_or_load()
        self.update()
        self.show()

    def save(self):
        import os
        import datetime

        path = self.p.working_directory+'/temp/clustering_tool_labels.pkl'
        if os.path.exists(path):
            dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            os.rename(path, path[:-4] + '_' + dt + '.pkl')

        with open(path, 'wb') as f:
            pickle.dump(self.data, f, -1)

    def select_all(self, key):
        self.grids[key].swap_selection()

    def select_until(self, key):
        self.grids[key].select_all_until_first()

    def move_selected_to(self, to_key='single'):
        for key, g in self.grids.iteritems():
            ids = g.get_selected()
            for id_ in ids:
                try:
                    self.data[key].remove(id_)
                except ValueError:
                    pass

                try:
                    self.undecided.remove(id_)
                except ValueError:
                    print "not present in undecided", id_

                self.data[to_key].append(id_)

            g.deselect_all()

        self.redraw_grids()

    def compute_or_load(self, first_run=True):
        try:
            with open(p.working_directory + '/temp/clustering.pkl') as f:
                up = pickle.Unpickler(f)
                up.load()
                self.vertices = up.load()
        except:
            if first_run:
                clustering(self.p)
                self.compute_or_load(first_run=False)
            else:
                raise Exception("loading failed...")

    def __controls(self):
        k = cv2.waitKey()

        if k == 115:
            return "single"
        elif k == 109:
            return "multi"
        elif k == 110:
            return "noise"
        elif k == 112:
            return "part"
        elif k == 27:
            return "exit"

    def load_data(self, compute):
        with open(p.working_directory + '/temp/clustering.pkl') as f:
            up = pickle.Unpickler(f)
            data = up.load()
            vertices = up.load()
            labels = up.load()

        scaler = StandardScaler()
        X = scaler.fit_transform(data)

        images = {}
        self.used_ids = set()
        if not compute:
            undecided = []
            try:
                with open(p.working_directory + '/temp/clustering_tool.pkl') as f:
                    images, undecided = pickle.load(f)
            except:
                compute = True
                print "clustering_tool.pkl NOT LOADED"

            try:
                with open(p.working_directory + '/temp/clustering_tool_labels.pkl') as f:
                    self.data = pickle.load(f)
            except:
                self.data = {'single': [],
                            'multi': [],
                            'noise': [],
                            'part': []}

            for k in self.data.iterkeys():
                self.used_ids = self.used_ids.union(set(self.data[k]))

            to_remove = []
            for id_ in undecided:
                if id_ in self.used_ids:
                    to_remove.append(id_)

            for id_ in to_remove:
                undecided.remove(id_)

        return X, vertices, undecided, images, compute

    def human_iloop_classification(self, compute=False):
        p = self.p

        X, vertices, undecided, images, compute = self.load_data(compute)
        vm = get_auto_video_manager(p)

        if compute:
            id_ = 0

            data = {'single': [],
                    'multi': [],
                    'noise': [],
                    'part': []}

            undecided = []
            images = {}

            d = None
            ask = True
            n = 1000
            for i in range(n):
                im = draw_region(p, vm, vertices[id_])
                if im.shape[0] == 0 or im.shape[1] == 0:
                    print p.gm.region(vertices[id_]).area(), vertices[id_]
                    continue

                print im.shape
                cv2.imshow('im', im)
                if ask:
                    key = self.__controls()

                    if key == 'exit':
                        ask = False
                    else:
                        data[key].append(id_)

                if not ask:
                    if id_ not in self.used_ids:
                        undecided.append(id_)

                images[id_] = im

                new_d = cdist([X[id_]], X)
                if d is None:
                    d = new_d
                else:
                    d = np.minimum(d, new_d)

                new_id = np.argmax(d)
                # if not enough data...
                if new_id == id_:
                    break

                id_ = new_id

                print_progress(i, n)

            with open(p.working_directory+'/temp/clustering_tool.pkl', 'wb') as f:
                pickle.dump((images, undecided), f)

        self.images = images
        self.undecided = undecided
        self.X = X
        self.redraw_grids()

    def active_features_vect(self):
        f_ch = [self.ch_area, self.ch_a, self.ch_b, self.ch_min_i, self.ch_max_i, self.ch_margin, self.ch_c_len]

        num_f = 7
        active = [False] * num_f
        for i in range(num_f):
            active[i] = f_ch[i].isChecked()

        return np.array(active)

    def redraw_grids(self):
        for g in self.grids.itervalues():
            self.vbox.removeWidget(g)

            g.setParent(None)

        self.singles = ImgGridWidget(cols=self.COLS, element_width=self.WW)
        self.multi = ImgGridWidget(cols=self.COLS, element_width=self.WW)
        self.noise = ImgGridWidget(cols=self.COLS, element_width=self.WW)
        self.part = ImgGridWidget(cols=self.COLS, element_width=self.WW)

        self.hbox.addWidget(self.singles)
        self.hbox.addWidget(self.multi)
        self.hbox.addWidget(self.noise)
        self.hbox.addWidget(self.part)

        self.grids = {'single': self.singles, 'multi': self.multi, 'noise': self.noise, 'part': self.part}

        active_f = self.active_features_vect()
        if self.show_decided.isChecked():
            for key, d in self.data.iteritems():
                if len(d) == 0:
                    continue

                for id_ in d:
                    if id_ not in self.images:
                        continue

                    item = self.make_item(self.images[id_], id_)
                    self.grids[key].add_item(item)

        items = {'single': [], 'multi': [], 'noise': [], 'part': []}
        if self.show_undecided.isChecked():
            for id_ in self.undecided:
                key, d_ = self.classify(id_, active_f)

                item = self.make_item(self.images[id_], id_)
                items[key].append((item, d_))

        for key, items in items.iteritems():
            items = sorted(items, key=lambda x: x[1])
            for it, _ in items:
                self.grids[key].add_item(it)

    def classify(self, id_, active_f):
        m = np.inf
        mk = 'single'
        for key, ids_ in self.data.iteritems():
            if len(ids_) == 0:
                continue

            dists_ = cdist([self.X[id_][active_f]], self.X[ids_][:, active_f])
            m_ = dists_.min()

            if m_ < m:
                m = m_
                mk = key

        return mk, m

    def make_item(self, im, id_):
        from PyQt4 import QtGui, QtCore
        from gui.gui_utils import SelectableQLabel
        from PIL import ImageQt


        im_ = np.zeros((max(im.shape[0], self.HH), max(im.shape[1], self.WW), 3), dtype=np.uint8)
        im_[:im.shape[0], :im.shape[1], :] = im
        im = im_

        img_q = ImageQt.QImage(im.data, im.shape[1], im.shape[0], im.shape[1] * 3, 13)
        pix_map = QtGui.QPixmap.fromImage(img_q.rgbSwapped())

        item = SelectableQLabel(id=id_)

        item.setScaledContents(True)
        if im.shape[0] > self.HH or im.shape[1] > self.WW:
            item.setFixedSize(self.HH, self.WW)

        item.setPixmap(pix_map)

        return item

    def train(self, n):
        with open(p.working_directory + '/temp/clustering_tool.pkl') as f:
            _, undecided = pickle.load(f)

        gt = self.data
        print "#S: {} #M: {} #N: {} #P: {}".format(len(gt['single']),
                                  len(gt['multi']),
                                  len(gt['noise']),
                                  len(gt['part']))
        gt_map = {}

        self.data = {'single': [],
                     'multi': [],
                     'noise': [],
                     'part': []}

        for key, d in gt.iteritems():
            for id_ in d:
                gt_map[id_] = key

        in_gt = set()
        i = 0
        # undecided are in order based on diferences
        for id_ in undecided:
            if id_ not in gt_map:
                continue

            key = gt_map[id_]
            self.data[key].append(id_)
            in_gt.add(id_)
            i += 1

            if i == n:
                break

        print "TRAINING SET: #S: {} #M: {} #N: {} #P: {}".format(len(self.data['single']),
                                                   len(self.data['multi']),
                                                   len(self.data['noise']),
                                                   len(self.data['part']))

        return gt_map, in_gt



    def eval(self, training_n=5):
        gt_map, in_gt = self.train(training_n)

        active_f = self.active_features_vect()

        mistakes = []
        correct = 0

        for id_, gt_c in gt_map.iteritems():
            if id_ in in_gt:
                continue
        # for id_ in undecided:
            c, d_ = self.classify(id_, active_f)

            if c == gt_map[id_]:
                correct += 1
            else:
                mistakes.append((id_, (c, d_, gt_map[id_])))

        print correct, len(mistakes)

        # self.data = {'single': [],
        #              'multi': [],
        #              'noise': [],
        #              'part': []}
        #
        # for id_, (c, _, _) in mistakes:
        #     self.data[c].append(id_)

        self.show_decided.setChecked(True)
        self.show_undecided.setChecked(False)
        self.redraw_grids()
        # self.data = gt

    def classify_project(self, p, data=None, train_n=30):
        from utils.gt.gt import GT
        gt = GT(num_ids = len(p.animals))
        gt.load(p.GT_file)

        gt.get_single_region_ids(p)

        if data is None:
            self.train(train_n)
        else:
            self.data = data

        active_f = self.active_features_vect()

        type_map = {'single': 0, 'multi': 1, 'noise': 2, 'part': 3}

        t_classes = {}
        for t in p.chm.chunk_gen():
            freq = [0, 0, 0, 0]
            for v in t.v_gen():
                c, d_ = self.classify(v, active_f)

                freq[type_map[c]] += 1

            gt_id = gt.tracklet_id_set(t, p)

            t_class = np.argmax(freq)

            if len(gt_id) == 1 and t_class != 0:
                print "ERROR, SINGLE not classified properly"

            if len(gt_id) == 0 and t_class != 2:
                print "ERROR, NOISE not classified properly"

            if len(gt_id) > 1 and t_class != 1:
                print "ERROR, MULTI not classified properly"

            t.segmentation_class = t_class

            t_classes[t.id()] = t_class
            print t.id(), t.length(), t_class, freq, "{:.2%}".format(freq[t_class] / float(np.sum(freq)))

        self.p.save_semistate('tracklets_s_classified')
        print "Classification DONE"




if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    from core.project.project import Project

    p = Project()
    wd = '/Users/flipajs/Documents/wd/FERDA/Cam1_playground'
    wd = '/Users/flipajs/Documents/wd/FERDA/Sowbug3'
    # wd = '/Users/flipajs/Documents/wd/FERDA/Camera3'
    wd = '/Users/flipajs/Documents/wd/FERDA/zebrafish_playground'
    p.load_semistate(wd, state='eps_edge_filter',
                     one_vertex_chunk=True, update_t_nodes=True)

    ex = ClusteringTool(p)
    ex.raise_()
    ex.activateWindow()

    ex.human_iloop_classification()

    ex.classify_project(p, train_n=30)
    # for n in [10, 30, 50, 100, 200]:
    # for n in [100]:
    #     ex.eval(training_n=n)
    #     ex.load_data(False)

    app.exec_()
    app.deleteLater()
    sys.exit()



