from PyQt6 import QtCore, QtGui, QtWidgets
import sys
import core.region.clustering
import numpy as np
from gui.img_grid.img_grid_widget import ImgGridWidget
from functools import partial
from core.config import config


class RegionClassifierTool(QtWidgets.QWizardPage):

    def __init__(self):
        super(RegionClassifierTool, self).__init__()

        self.clustering = core.region.clustering.RegionCardinality()
        self.samples = []
        self.labeled_samples = []
        self.gather_samples_thread = None

        # build UI
        self.vbox = QtWidgets.QVBoxLayout()
        self.setLayout(self.vbox)
        self.hbox = QtWidgets.QHBoxLayout()
        self.vbox.addLayout(self.hbox)

        self.GRID_ITEM_WH = (120, 120)
        grids_desc = [
            {'key': 'single',
             'label': 'single-ID'
             },
            {'key': 'multi',
             'label': 'multi-ID'
             },
            {'key': 'noise',
             'label': 'no-ID'
             },
            {'key': 'part',
             'label': 'ID-part'
             },
        ]
        self.grids = {}
        for desc in grids_desc:
            self.grids[desc['key']] = ImgGridWidget(cols=2, element_width=self.GRID_ITEM_WH[0])
            widget = QtWidgets.QWidget()
            widget.setLayout(QtWidgets.QVBoxLayout())
            widget.layout().addWidget(QtWidgets.QLabel(desc['label']))
            widget.layout().addWidget(self.grids[desc['key']])
            self.hbox.addWidget(widget)

        self.hbox_check = QtWidgets.QHBoxLayout()
        self.vbox.addLayout(self.hbox_check)
        # self.hbox_load_controls = QtGui.QHBoxLayout()
        # self.vbox.addLayout(self.hbox_load_controls)

        self.hbox_buttons = QtWidgets.QHBoxLayout()
        self.vbox.addLayout(self.hbox_buttons)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setMaximum(config['region_classifier']['samples_preselection_num'])
        self.vbox.addWidget(self.progress_bar)

        actions = [
            {'text': 'mark single-ID',
             'trigger': partial(self.move_selected_to, 'single'),
             'shortcut': QtCore.Qt.Modifier.SHIFT | QtCore.Qt.Key.Key_S
             },
            {'text': 'mark multi-ID',
             'trigger': partial(self.move_selected_to, 'multi'),
             'shortcut': QtCore.Qt.Modifier.SHIFT | QtCore.Qt.Key.Key_M
             },
            {'text': 'mark no-ID',
             'trigger': partial(self.move_selected_to, 'noise'),
             'shortcut': QtCore.Qt.Modifier.SHIFT | QtCore.Qt.Key.Key_N
             },
            {'text': 'mark part',
             'trigger': partial(self.move_selected_to, 'part'),
             'shortcut': QtCore.Qt.Modifier.SHIFT | QtCore.Qt.Key.Key_P
             },
            {'trigger': partial(self.move_selected_to, 'undecided'),
             'shortcut': QtCore.Qt.Modifier.SHIFT | QtCore.Qt.Key.Key_U
             },

            {'trigger': partial(self.select_all, 'single'),
             'shortcut': QtCore.Qt.Modifier.SHIFT | QtCore.Qt.Modifier.CTRL | QtCore.Qt.Key.Key_S
             },
            {'trigger': partial(self.select_all, 'multi'),
             'shortcut': QtCore.Qt.Modifier.SHIFT | QtCore.Qt.Modifier.CTRL | QtCore.Qt.Key.Key_M
             },
            {'trigger': partial(self.select_all, 'noise'),
             'shortcut': QtCore.Qt.Modifier.SHIFT | QtCore.Qt.Modifier.CTRL | QtCore.Qt.Key.Key_N
             },
            {'trigger': partial(self.select_all, 'part'),
             'shortcut': QtCore.Qt.Modifier.SHIFT | QtCore.Qt.Modifier.CTRL | QtCore.Qt.Key.Key_P
             },

            {'trigger': partial(self.select_until, 'single'),
             'shortcut': QtCore.Qt.Modifier.CTRL | QtCore.Qt.Key.Key_S
             },
            {'trigger': partial(self.select_until, 'multi'),
             'shortcut': QtCore.Qt.Modifier.CTRL | QtCore.Qt.Key.Key_M
             },
            {'trigger': partial(self.select_until, 'noise'),
             'shortcut': QtCore.Qt.Modifier.CTRL | QtCore.Qt.Key.Key_N
             },
            {'trigger': partial(self.select_until, 'part'),
             'shortcut': QtCore.Qt.Modifier.CTRL | QtCore.Qt.Key.Key_P
             },
        ]

        for a in actions:
            action = QtGui.QAction(self)
            action.triggered.connect(a['trigger'])
            action.setShortcut(QtGui.QKeySequence(a['shortcut']))
            self.addAction(action)
            if 'text' in a:
                button = QtWidgets.QPushButton(a['text'])
                button.clicked.connect(a['trigger'])
                button.setToolTip(action.shortcut().toString())
                self.hbox_buttons.addWidget(button)

        # CHECKBOXES
        self.show_decided = QtWidgets.QCheckBox('show decided')
        self.show_decided.setChecked(False)
        self.show_decided.stateChanged.connect(self.redraw_grids)
        self.hbox_buttons.addWidget(self.show_decided)

        self.show_undecided = QtWidgets.QCheckBox('show undecided')
        self.show_undecided.setChecked(True)
        self.show_undecided.stateChanged.connect(self.redraw_grids)
        self.hbox_buttons.addWidget(self.show_undecided)

        self.hbox_check.addWidget(QtWidgets.QLabel('feature space: '))
        self.feature_checkboxes = []
        for i, description in enumerate(core.region.clustering.region_features.keys()):
            w = QtWidgets.QCheckBox(description)
            if description in config['region_classifier']['default_features']:
                w.setChecked(True)
            else:
                w.setChecked(False)
            w.stateChanged.connect(self.active_features_changed)
            self.hbox_check.addWidget(w)
            self.feature_checkboxes.append(w)
        self.apply_active_features()

        self.setDisabled(True)

    def initializePage(self):
        class GatherSamplesThread(QtCore.QThread):
            samples_ready = QtCore.pyqtSignal(list)
            update = QtCore.pyqtSignal(int)

            def __init__(self, project, clustering):
                super(GatherSamplesThread, self).__init__()
                self.project = project
                self.clustering = clustering

            def run(self):
                samples = self.clustering.gather_diverse_samples(
                    config['region_classifier']['samples_preselection_num'],
                    config['region_classifier']['samples_num'],
                    self.project,
                    self.update.emit
                )
                self.samples_ready.emit(samples)

            def __del__(self):
                self.wait()

        self.gather_samples_thread = GatherSamplesThread(self.wizard().project, self.clustering)
        self.gather_samples_thread.samples_ready.connect(self.retrieve_samples)
        self.gather_samples_thread.update.connect(self.progress_bar_update)
        self.gather_samples_thread.finished.connect(self.gather_samples_finished)
        self.gather_samples_thread.start()

    def validatePage(self):
        self.wizard().project.region_cardinality_classifier = self.clustering
        self.wizard().project.save()
        return True

    def progress_bar_update(self, delta):
        self.progress_bar.setValue(self.progress_bar.value() + delta)

    def retrieve_samples(self, samples):
        self.samples = samples
        self.clustering.init_scaler(self.samples)
        for i, s in enumerate(self.samples):
            s.widget = self.make_item(s.image, i)
            s.label = 'single'
        self.redraw_grids()

    def gather_samples_finished(self):
        self.setEnabled(True)
        self.progress_bar.setValue(config['region_classifier']['samples_preselection_num'])

    def apply_active_features(self):
        mask = np.zeros(len(self.feature_checkboxes), dtype=np.int)
        for i, checkbox in enumerate(self.feature_checkboxes):
            if checkbox.checkState() == QtCore.Qt.CheckState.Checked:
                mask[i] = True
        self.clustering.set_active_features(mask)

    def active_features_changed(self):
        self.apply_active_features()
        self.clustering.train(self.labeled_samples)
        self.samples = self.clustering.classify_samples(self.samples)
        self.redraw_grids()

    def select_all(self, key):
        self.grids[key].swap_selection()

    def select_until(self, key):
        self.grids[key].select_all_until_first()

    def move_selected_to(self, to):
        for label, grid in list(self.grids.items()):
            for item in grid.get_selected_items():
                if to != 'undecided':
                    item.label = to
                    if item in self.samples:
                        self.samples.remove(item)
                    elif item in self.labeled_samples:
                        self.labeled_samples.remove(item)
                    else:
                        assert True
                    self.labeled_samples.append(item)
                else:
                    if item in self.labeled_samples:
                        self.labeled_samples.remove(item)
                        self.samples.append(item)
                    else:
                        assert item in self.samples

            grid.deselect_all()

        self.clustering.train(self.labeled_samples)
        if self.samples:
            self.samples = self.clustering.classify_samples(self.samples)
        self.redraw_grids()

    def redraw_grids(self):
        for item in list(self.grids.values()):
            item.delete_all()

        samples = []
        if self.show_decided.isChecked():
            samples.extend(self.labeled_samples)

        if self.show_undecided.isChecked():
            samples.extend(self.samples)

        # add samples of corresponding lables to ImgGridWidgets
        for label in list(self.grids.keys()):
            samples_label = [s for s in samples if s.label == label]
            for i, s in enumerate(sorted(samples_label, reverse=True)):
                s.widget = self.make_item(s.image, i)
                self.grids[label].add_item(s)

    def make_item(self, im, item_id):
        from PyQt6 import QtGui, QtWidgets
        from gui.gui_utils import SelectableQLabel
        from PIL import ImageQt

        im_item = np.zeros((max(im.shape[0], self.GRID_ITEM_WH[1]), max(im.shape[1], self.GRID_ITEM_WH[0]), 3), dtype=np.uint8)
        im_item[:im.shape[0], :im.shape[1], :] = im

        img_q = ImageQt.QImage(im_item.data, im_item.shape[1], im_item.shape[0], im_item.shape[1] * 3, 13)
        pix_map = QtGui.QPixmap.fromImage(img_q.rgbSwapped())

        item = SelectableQLabel(id=item_id)
        item.setScaledContents(True)
        item.setFixedSize(self.GRID_ITEM_WH[0], self.GRID_ITEM_WH[1])
        item.setPixmap(pix_map)
        return item


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    from core.project.project import Project

    p = Project()
    wd = '/home/matej/prace/ferda/projects/Sowbug_deleteme'
    # # wd = '/Users/flipajs/Documents/wd/FERDA/Cam1_rf'
    # # wd = '/Users/flipajs/Documents/wd/FERDA/Sowbug3'
    # # wd = '/Users/flipajs/Documents/wd/FERDA/Camera3'
    # # wd = '/Users/flipajs/Documents/wd/FERDA/zebrafish_playground'
    p.load(wd)
    ex = RegionClassifierTool(p)
    ex.raise_()
    ex.activateWindow()
    app.exec_()
    app.deleteLater()
