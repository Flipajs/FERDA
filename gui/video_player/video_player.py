import operator

import warnings
from PyQt5 import QtCore, QtGui, QtWidgets

from gui.gui_utils import SelectAllLineEdit
from gui.img_controls.gui_utils import cvimg2qtpixmap
from gui.img_controls.my_view import MyView
from gui.settings import Settings as S_
from utils.video_manager import get_auto_video_manager


class VideoPlayer(QtWidgets.QWidget):
    # TODO:
    _play_forward = True
    _scene = None
    _video_step = 1
    _PERMANENT_VISUALISATION_Z_LVL = 1.0
    _looper = None

    def __init__(self, project, frame_change_callback=None, image_processor_callback=None):
        """
        image_processor_callback will be called on every frame. img = image_processor_callback(img)

        Args:
            video_manager:
            frame_change_callback:
            image_processor_callback:
        """
        super(VideoPlayer, self).__init__()

        self.setLayout(QtWidgets.QVBoxLayout())

        self._vm = get_auto_video_manager(project)
        self._frame_change_callback = frame_change_callback
        self._image_processor_callback = image_processor_callback

        self._view = MyView()
        self._scene = QtWidgets.QGraphicsScene(self._view)
        self._view.setScene(self._scene)
        self.layout().addWidget(self._view)

        self._scene_items = {'temp': {}, 'permanent': {}}

        self.frame_rate = 30
        self.timer = QtCore.QTimer()
        self.timer.setInterval(1000 // self.frame_rate)

        self._video_controls()
        self._add_actions()

        next(self)
        self.updateGeometry()

    def set_frame_change_callback(self, frame_change_callback):
        self._frame_change_callback = frame_change_callback

    def set_image_processor_callback(self, image_processor_callback):
        self._image_processor_callback = image_processor_callback

    def _video_controls(self):
        from .video_slider import VideoSlider

        self.video_control_widget = QtWidgets.QWidget()
        self.video_control_layout = QtWidgets.QVBoxLayout()
        self.video_control_widget.setLayout(self.video_control_layout)
        self.video_control_widget.setMaximumHeight(70)
        self.video_control_widget.setContentsMargins(0, 0, 0, 0)

        self.video_control_buttons_widget = QtWidgets.QWidget()
        self.video_control_buttons_layout = QtWidgets.QHBoxLayout()
        self.video_control_buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.video_control_buttons_widget.setLayout(self.video_control_buttons_layout)

        self.video_slider = VideoSlider()
        self.video_slider.setOrientation(QtCore.Qt.Horizontal)
        self.video_slider.setFocusPolicy(QtCore.Qt.NoFocus)
        self.video_slider.setMaximumHeight(15)
        self.video_slider.setMaximum(self._vm.total_frame_count())

        self.video_control_layout.addWidget(self.video_slider)
        self.video_control_layout.addWidget(self.video_control_buttons_widget)

        self.layout().addWidget(self.video_control_widget)

        self.speedSlider = QtWidgets.QSlider()
        self.speedSlider.setOrientation(QtCore.Qt.Horizontal)
        self.speedSlider.setMinimum(0)
        self.speedSlider.setMaximum(99)

        self.backward = QtWidgets.QPushButton('<')
        self.backward.setShortcut(S_.controls.video_prev)
        self.playPause = QtWidgets.QPushButton('play')
        self.playPause.setShortcut(S_.controls.video_play_pause)
        self.forward = QtWidgets.QPushButton('>')
        self.forward.setShortcut(S_.controls.video_next)
        self.frame_edit = SelectAllLineEdit()
        self.frame_edit.returnPressed.connect(self.goto)
        self.frame_edit.setFixedHeight(30)

        self.fpsLabel = QtWidgets.QLabel()
        self.fpsLabel.setAlignment(QtCore.Qt.AlignRight)

        self.frame_jump_button = QtWidgets.QPushButton('go')
        self.frame_jump_button.clicked.connect(self.goto)

        self.frame_jump_button.setFocusPolicy(QtCore.Qt.StrongFocus)

        self.visu_controls_layout = QtWidgets.QHBoxLayout()
        self.video_control_buttons_layout.addLayout(self.visu_controls_layout)
        self.video_control_buttons_layout.addWidget(self.speedSlider)

        self.video_step_label = QtWidgets.QLabel('1')

        self.video_control_buttons_layout.addWidget(self.speedSlider)
        self.video_control_buttons_layout.addWidget(self.fpsLabel)
        self.video_control_buttons_layout.addWidget(self.video_step_label)
        self.video_control_buttons_layout.addWidget(self.backward)
        self.video_control_buttons_layout.addWidget(self.playPause)
        self.video_control_buttons_layout.addWidget(self.forward)
        self.video_control_buttons_layout.addWidget(self.frame_edit)
        self.video_control_buttons_layout.addWidget(self.frame_jump_button)

        self.init_speed_slider()
        self.connect_GUI()

    def _add_actions(self):
        #   video step
        self.increase_video_step_a = QtWidgets.QAction('increase video step', self)
        self.increase_video_step_a.triggered.connect(lambda x: self.increase_video_step())
        self.increase_video_step_a.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_2))
        self.addAction(self.increase_video_step_a)

        self.decrease_video_step_a = QtWidgets.QAction('decrease video step', self)
        self.decrease_video_step_a.triggered.connect(lambda x: self.decrease_video_step())
        self.decrease_video_step_a.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_1))
        self.addAction(self.decrease_video_step_a)

        self.small_video_forward_a = QtWidgets.QAction('small next', self)
        self.small_video_forward_a.triggered.connect(lambda x: self.goto(self._vm.frame_number() + 3))
        self.small_video_forward_a.setShortcut(QtGui.QKeySequence(QtCore.Qt.ALT + QtCore.Qt.Key_N))
        self.addAction(self.small_video_forward_a)

        self.small_video_backward_a = QtWidgets.QAction('small back', self)
        self.small_video_backward_a.triggered.connect(lambda x: self.goto(self._vm.frame_number() - 3))
        self.small_video_backward_a.setShortcut(QtGui.QKeySequence(QtCore.Qt.ALT + QtCore.Qt.Key_B))
        self.addAction(self.small_video_backward_a)

        self.middle_video_forward_a = QtWidgets.QAction('middle next', self)
        self.middle_video_forward_a.triggered.connect(lambda x: self.goto(self._vm.frame_number() + 10))
        self.middle_video_forward_a.setShortcut(QtGui.QKeySequence(QtCore.Qt.SHIFT + QtCore.Qt.Key_N))
        self.addAction(self.middle_video_forward_a)

        self.middle_video_backward_a = QtWidgets.QAction('middle back', self)
        self.middle_video_backward_a.triggered.connect(lambda x: self.goto(self._vm.frame_number() - 10))
        self.middle_video_backward_a.setShortcut(QtGui.QKeySequence(QtCore.Qt.SHIFT + QtCore.Qt.Key_B))
        self.addAction(self.middle_video_backward_a)

        self.big_video_forward_a = QtWidgets.QAction('big next', self)
        self.big_video_forward_a.triggered.connect(lambda x: self.goto(self._vm.frame_number() + 50))
        self.big_video_forward_a.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_0))
        self.addAction(self.big_video_forward_a)

        self.big_video_backward_a = QtWidgets.QAction('big back', self)
        self.big_video_backward_a.triggered.connect(lambda x: self.goto(self._vm.frame_number() - 50))
        self.big_video_backward_a.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_9))
        self.addAction(self.big_video_backward_a)

    def connect_GUI(self):
        """Connects GUI elements to appropriate methods"""
        self.forward.clicked.connect(self.__next__)
        self.backward.clicked.connect(self.prev)
        self.playPause.clicked.connect(self.play_pause)
        self.speedSlider.valueChanged.connect(self.speed_slider_changed)
        self.video_slider.valueChanged.connect(self.video_slider_changed)
        self.timer.timeout.connect(self.__next__)

    def play_pause(self):
        if self.timer.isActive():
            self.pause()
        else:
            self.play()

    def play(self):
        self.timer.start()
        self.playPause.setText("pause")
        self.playPause.setShortcut(S_.controls.video_play_pause)

    def pause(self):
        self.timer.stop()
        self.playPause.setText("play")
        self.playPause.setShortcut(S_.controls.video_play_pause)

    def reverse_playhead(self):
        self._play_forward = not self._play_forward

    def play_reversed(self):
        self._play_forward = False

    def _get_next_operator(self):
        if self._play_forward:
            return operator.add
        else:
            return operator.sub

    def _get_prev_operator(self):
        if self._play_forward:
            return operator.sub
        else:
            return operator.add

    def redraw_visualisations(self):
        self.clear_all_temp_visualisations()
        self._update_bg(self._vm.img())
        self._frame_change_callback()

    def _change_frame(self, operator=operator.add, frame=None):
        self.clear_all_temp_visualisations()
        # TODO: if looper?
        # try:
        if frame is None:
            frame = operator(self._vm.frame_number(), self._video_step)

        # TODO: maybe allow cycling in future?
        if frame < 0:
            frame = 0
        if frame >= self._vm.total_frame_count():
            frame = self._vm.total_frame_count() - 1

        img = self._vm.get_frame(frame)

        if img is None:
            warnings.warn("img is None")
            if frame != self._vm.total_frame_count():
                warnings.warn("cannot read frame: "+frame)
        else:
            self._update_bg(img)
            if self._frame_change_callback:
                self._frame_change_callback()

            self.update_frame_number()
        # except Exception as e:
        #     warnings.warn('EXCEPTION catched: '+str(e))

    def current_frame(self):
        return self._vm.frame_number()

    def total_frame_count(self):
        return self._vm.total_frame_count()

    def remove_items_category(self, type='temp', category=''):
        except_for = set(self._scene_items[type])
        if category in except_for:
            except_for.remove(category)
        else:
            warnings.warn('category: '+category+' doesn\'t exist')

        if type == 'permanent':
            warnings.warn('remove all permanent not implemented yet', UserWarning)
        else:
            self.clear_all_temp_visualisations(except_for=except_for)

    def _update_bg(self, img):
        if self._image_processor_callback:
            img = self._image_processor_callback(img)

        self.clear_all_temp_visualisations()
        pixmap = cvimg2qtpixmap(img)
        self.visualise_temp(QtWidgets.QGraphicsPixmapItem(pixmap), 'bg')

    def __next__(self):
        self._change_frame(self._get_next_operator())

    def prev(self):
        self._change_frame(self._get_prev_operator())

    def goto(self, frame=None):
        if frame is None:
            frame = int(self.frame_edit.text())

        if frame < 0:
            frame = 0

        if frame >= self._vm.total_frame_count():
            frame = self._vm.total_frame_count() - 1
            
        self._change_frame(frame=frame)
        self.setFocus()

    @property
    def video_step(self):
        return self._video_step

    @video_step.setter
    def video_step(self, value):
        self._video_step = value
        if self._video_step < 1:
            self._video_step = 1

        self.video_step_label.setText(str(self._video_step))

    def increase_video_step(self, value=1):
        self.video_step = self._video_step + value

    def decrease_video_step(self, value=1):
        self.video_step = self._video_step - value

    def visualise_temp(self, item, category='others'):
        self._scene.addItem(item)
        self._scene_items['temp'].setdefault(category, [])
        self._scene_items['temp'][category].append(item)

    def visualise_permanent(self, obj):
        # TODO: implement
        warnings.warn("not implemented yet", UserWarning)

    def clear_all_temp_visualisations(self, except_for=[]):
        delete_ = []
        for key, arr in self._scene_items['temp'].items():
            if key in except_for:
                continue

            for item in arr:
                self._scene.removeItem(item)

            delete_.append(key)

        for key in delete_:
            del self._scene_items['temp'][key]

    def init_speed_slider(self):
        """Initiates components associated with speed of viewing videos"""
        self.speedSlider.setValue(self.frame_rate)
        self.timer.setInterval(1000 // self.frame_rate)
        self.fpsLabel.setText(str(self.frame_rate) + ' fps')
        self.speedSlider.setMinimum(1)
        self.speedSlider.setMaximum(120)

    def speed_slider_changed(self):
        """Method invoked when value of slider controlling speed of video changed it's value"""
        self.frame_rate = self.speedSlider.value()
        self.timer.setInterval(1000 // self.frame_rate)
        self.fpsLabel.setText(str(self.frame_rate) + ' fps')

    def video_slider_changed(self):
        """Method invoked when slider controlling video position changed. To differentiate between
        situations when the slider was changed by user or by part of the program, use videoSlider.usercontrolled
        and videoSlider.recentlyreleased
        """
        if self.video_slider.usercontrolled:
            self.goto(self.video_slider.value())
        elif self.video_slider.recentlyreleased:
            self.video_slider.recentlyreleased = False
            self.goto(self.video_slider.value())

    def update_frame_number(self):
        """Updates values of components displaying frame number"""
        s = str(int(self._vm.frame_number() + 1)) + '/' + str(self._vm.total_frame_count())
        self.frame_edit.setText(s)
        self.video_slider.setValue(self._vm.frame_number())


if __name__ == '__main__':
    vp = VideoPlayer(None)
    vp.play_reversed()
    print("TEST")
