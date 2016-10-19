

class VideoPlayer:
    def __init__(self, video_manager, frame_change_callback=None):
        self.vm = video_manager
        self.video_step_ = 1
        # TODO:
        self.play_forward_ = True

        # TODO:
        self.qt_scene = None
        self.frame_change_callback = frame_change_callback

    def play(self):
        pass

    def pause(self):
        pass

    def play_reversed(self):
        self.play_forward_ = False
        # TODO: remove when functionality is implemented
        raise Exception("play_reversed not implemented yet")

    def next(self):
        pass

    def prev(self):
        pass

    def goto(self, frame):
        pass

    @property
    def video_step(self):
        return self.video_step_

    @video_step.setter
    def video_step(self, value):
        self.video_step_ = value
        if self.video_step_ < 1:
            self.video_step_ = 1

    def increase_video_step(self, value=1):
        self.video_step(self.video_step_ + value)

    def decrease_video_step(self, value=1):
        self.video_step(self.video_step_ - value)

