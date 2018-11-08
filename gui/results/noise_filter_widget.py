from __future__ import unicode_literals
from PyQt4 import QtGui

from gui.img_grid.img_grid_widget import ImgGridWidget
from gui.results.noise_filter_computer import NoiseFilterComputer


class NoiseFilterWidget(QtGui.QWidget):
    def __init__(self, project, steps, elem_width, cols):
        super(NoiseFilterWidget, self).__init__()

        self.project = project
        self.steps = steps

        self.vbox = QtGui.QVBoxLayout()
        self.setLayout(self.vbox)

        self.elem_width = elem_width
        self.noise_nodes_widget = ImgGridWidget()
        self.noise_nodes_widget.reshape(cols, elem_width)

        self.progress_bar = QtGui.QProgressBar()
        self.progress_bar.setRange(0, 0)

        self.threshold = QtGui.QDoubleSpinBox()
        self.threshold.setMinimum(0)
        self.threshold.setMaximum(1.0)
        self.threshold.setSingleStep(0.01)
        self.threshold.setValue(project.solver_parameters.antlikeness_threshold)

        self.hbox = QtGui.QHBoxLayout()
        self.hbox.addWidget(self.threshold)
        self.run_button = QtGui.QPushButton('run...')
        self.run_button.clicked.connect(self.run_noise_filter)
        self.hbox.addWidget(self.run_button)

        self.vbox.addLayout(self.hbox)
        self.vbox.addWidget(self.progress_bar)
        self.vbox.addWidget(self.noise_nodes_widget)

        self.noise_nodes_confirm_b = QtGui.QPushButton('remove selected')
        self.noise_nodes_confirm_b.clicked.connect(self.remove_noise)
        self.vbox.addWidget(self.noise_nodes_confirm_b)

        # TODO:
        # self.undo_action = QtGui.QAction('undo', self)
        # self.undo_action.triggered.connect(self.undo)
        # self.undo_action.setShortcut(S_.controls.undo)
        # self.addAction(self.undo_action)


    def run_noise_filter(self):
        th = self.threshold.value()
        self.thread = NoiseFilterComputer(self.project.solver, self.project, self.steps, th)
        self.thread.part_done.connect(self.noise_part_done_)
        self.thread.proc_done.connect(self.noise_finished_)
        self.thread.set_range.connect(self.progress_bar.setMaximum)
        self.thread.start()


    def remove_noise(self):
        # TODO: add actions
        to_remove = self.noise_nodes_widget.get_selected()
        affected = []
        for v in to_remove:
            # if n in self.solver.g:
            affected.extend(self.project.solver.strong_remove(v))

        to_confirm = self.noise_nodes_widget.get_unselected()

        # TODO: add some flag that the antlikeness is confirmed
        # for n in to_confirm:
        #     if n in self.project.solver.g:
        #         self.project.solver.g.node[n]['antlikeness'] = 1.0

        self.noise_nodes_widget.hide()

        self.project.solver.simplify(queue=affected, rules=[self.project.solver.update_costs])
        self.project.solver.simplify(queue=affected, rules=[self.project.solver.adaptive_threshold])
        # self.project.solver.simplify(rules=[self.project.solver.adaptive_threshold, self.project.solver.update_costs])
        # self.next_case()

    def noise_part_done_(self, val, img, region, vertex):
        from gui.gui_utils import get_img_qlabel

        self.progress_bar.setValue(val)
        item = get_img_qlabel(region.pts(), img, vertex, self.elem_width, self.elem_width, filled=True)
        item.set_selected(True)
        self.noise_nodes_widget.add_item(item)

    def noise_finished_(self):
        self.progress_bar.setParent(None)
        self.noise_nodes_confirm_b.show()