import warnings
from PyQt4 import QtCore, QtGui

from gui.img_controls.my_view import MyView


class VideoPlayer(QtGui.QWidget):
    # TODO:
    _play_forward = True
    # TODO:
    _scene = None
    _video_step = 1
    _PERMANENT_VISUALISATION_Z_LVL = 1.0

    def __init__(self, video_manager, frame_change_callback=None):
        super(VideoPlayer, self).__init__()

        self._vm = video_manager
        self._frame_change_callback = frame_change_callback

        self._scene = QtGui.QGraphicsScene()

        self._view = MyView(self)
        self._view.setScene(self._scene)

        self.next()

    def play(self):
        pass

    def pause(self):
        pass

    def play_reversed(self):
        self._play_forward = False
        # TODO: remove when functionality is implemented
        warnings.warn("not implemented yet", UserWarning)

    def next(self):
        pass

    def prev(self):
        pass

    def goto(self, frame):
        pass

    @property
    def video_step(self):
        return self._video_step

    @video_step.setter
    def video_step(self, value):
        self._video_step = value
        if self._video_step < 1:
            self._video_step = 1

    def increase_video_step(self, value=1):
        self.video_step(self._video_step + value)

    def decrease_video_step(self, value=1):
        self.video_step(self._video_step - value)

    def visualise_temp(self, obj, lvl=-1):
        # TODO: add to scene
        # TODO: register to remove in next step...
        pass

    def visualise_permanent(self, obj):
        # TODO: implement
        warnings.warn("not implemented yet", UserWarning)

    def clear_all_visualisations(self):
        # TODO: go through all items and remove them (except for background)
        pass





if __name__ == '__main__':
    vp = VideoPlayer(None)
    vp.play_reversed()
    print "TEST"
