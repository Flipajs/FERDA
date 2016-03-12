from PyQt4 import QtGui, QtCore
from matplotlib import colors
from matplotlib import cm as cmx

import sys
import math
import random
from core.graph.region_chunk import RegionChunk

# TODO: opravit chybu databaze pri nacitani projektu rucne (spatne vlakno)

class ColorManager():
    def __init__(self, length, limit, overlap=30, mode="rand", cmap='Accent', rand_quality=80, rand_loop_limit=100):
        """
        :param length: the length of the video (frames)
        :param limit: the max number of colors to be used. this is crucial in cmap and rainbow mode.
        :param overlap: minimum distance between two similar colors in frames. Should be positive int.
        :param mode: "cmap", "rainbow" or "rand" (default), chooses the colors randomly or from a cmap
        :param cmap: the cmap to be used in "cmap" mode
        :param rand_quality: the higher the quality, the bigger the difference between colors (but also loops required).
                             Values smaller than default 80 can produce very similar colors
        :param rand_loop_limit: number of tries until color manager gives up finding ideal color for a track and chooses
                             entirely at random. Must be positive int.
        """

        # TODO: http://llllll.li/randomColor/ has a distinguishable color generator on his todo list, check it later
        # UPDATE: He didn't do it yet..
        # list to store all the tracks
        self.tracks = []

        #lenght of the video
        self.length = length

        # max count of colors
        self.limit = limit

        self.overlap = overlap

        if mode == "cmap":
            self.mode = "cmap"
            color_norm = colors.Normalize(vmin=0, vmax=limit)
            self.scalar_map = cmx.ScalarMappable(norm=color_norm, cmap=cmap)
        elif mode == "rainbow":
            # http://stackoverflow.com/questions/470690/how-to-automatically-generate-n-distinct-colors
            self.mode = "rainbow"
            self.colors_list = []
            if limit >= 2:
                dx = 1 / (limit - 1.0)
                for i in range (0, limit):
                    self.colors_list.append(self.generate_color_cube(i * dx))
            self.cube_id = 0
        else:
            self.mode = "rand"
            self.adjacency = {}
            self.bg_color = QtGui.QColor().fromRgb(0, 0, 0)
            self.rand_quality = rand_quality
            self.rand_loop_limit = rand_loop_limit

        random.seed()
        self.id = 0

    def get_next_id(self):
        # return current id and raise it by one
        self.id += 1
        return self.id - 1

    def new_track(self, start, stop):
        """
        Adds a new track to the color manager
        :param start: the first frame of the track
        :param stop: the last frame
        :return: tuple: (color, id)
        """

        # create a new track
        track = Track(start, stop, self.get_next_id(), QtGui.QColor().fromRgb(0, 0, 0))

        # find a suitable color for it
        if self.mode == "cmap":
            color = self.find_color_cmap(track)
        elif self.mode == "rainbow":
            color = self.find_color_cube()
        else:
            self.adjacency[track.id] = []
            for t in self.tracks:
                if self.collide(t, track) > 0:
                    self.adjacency[track.id].append(t)
                    self.adjacency[t.id].append(track)
            color = self.find_color_rand(track)
        track.set_color(color)

        # add it in the tracks list
        self.tracks.append(track)

        return color, self.id

    def delete_(self, id):
        """
        Delete the track from CM. To merge two tracks, use ColorManager.merge().
        :param id:
        :return: None
        """
        self.tracks.remove(self.find_by_id(id))

    def merge(self, id1, id2):
        """
        Merges two tracks into the first one. All free space between the tracks is added to the first track, too. The
        new track will have the color and id of the first track.
        :param id1: id of the first track
        :param id2: id of the second one
        :return: None
        """

        t1 = self.find_by_id(id1)
        t2 = self.find_by_id(id2)

        if t1.start > t2.start:
            t1.start = t2.start
        if t1.stop < t2.stop:
            t1.stop = t2.stop

        self.tracks.remove(t2)

    def find_by_id(self, id):
        for track in self.tracks:
            if track.id == id:
                return track

    def collide(self, track1, track2):
        """
        returns the length of the intersection of track1 and track2 (how long they "exist" together)
        :param track1: the first track
        :param track2: the second track (not necessarily in this order)
        :return:
        """
        start1 = track1.start
        stop1 = track1.stop + self.overlap
        start2 = track2.start
        stop2 = track2.stop + self.overlap

        # no overlap
        if (start1 <= start2 and stop1 <= start2)\
                or (start2 <= start1 and stop2 <= start1):
            return 0

        # 1:    --------
        # 2:          --------
        if start1 <= start2 and stop1 <= stop2:
            return abs(stop1 - start2)

        # 1:          --------
        # 2:    --------
        if start2 <= start1 and stop2 <= stop1:
            return abs(stop2 - start1)

        # 1:    ------------
        # 2:       ------
        if start1 <= start2 and stop1 >= stop2:
            return abs(track2.len)


        # 1:       ------
        # 2:    ------------
        if start2 <= start1 and stop2 >= stop1:
            return abs(track1.len)

        # in case of emergency (the impossible mistake)
        return 0

    def find_color_rand(self, track):
        counter = 0
        ok = False
        while not ok:
            ok = True
            # try to pick a color
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            c = QtGui.QColor.fromRgb(r, g, b)

            # give up after several unsuccessful loops
            if counter >= self.rand_loop_limit:
                # print "No color found!"
                # return last color
                return c
            counter += 1

            if self.get_yuv_distance(c, self.bg_color) <= self.rand_quality:
                ok = False
                continue

            for t in self.adjacency[track.id]:
                if self.get_yuv_distance(c, t.get_color()) <= self.rand_quality:
                    ok = False
                    break
        # print "(%s, %s, %s)" % (r, g, b)
        return c

    def find_color_cmap(self, track):
        while(True):
            ok = True
            # try to pick a random color position
            i = random.randint(0, self.limit)
            for t in self.tracks:
                # if a colliding track already uses it, try again
                if self.collide(track, t) > 0 and i == t.color_id:
                    i = (i + 1) % self.limit
                    ok = False
                    break
            if ok:
                tmp_color = self.scalar_map.to_rgba(i)
                track.set_color_id(i)
                return QtGui.QColor().fromRgbF(tmp_color[0], tmp_color[1], tmp_color[2])

    def find_color_cube(self):
        next_color = self.colors_list[self.cube_id]
        self.cube_id += 1
        print next_color
        return QtGui.QColor.fromRgbF(next_color[0], next_color[1], next_color[2])

    def generate_color_cube(self, x):
        r, g, b = 0, 0, 1
        if 0 <= x < 0.2:
            x = x / 0.2
            r = 0
            g = x
            b = 1
        elif x < 0.4:
            x = (x - 0.2) / 0.2
            r = 0
            g = 1
            b = 1
        elif x < 0.6:
            x = (x - 0.4) / 0.2
            r = x
            g = 1
            b = 0
        elif x < 0.8:
            x = (x - 0.6) / 0.2
            r = 1.0
            g = 1.0 - x
            b = 0.0
        elif x <= 1.0:
            x = (x - 0.8) / 0.2
            r = 1
            g = 0
            b = x
        return [r, g, b]

    def get_yuv_distance(self, c1, c2):
        # returns the "distance" of two colors in the YUV color system, which corresponds best to human perception
        # formulas to convert RGB to YUV (from wikipedia)
        y1 =  0.299 * c1.red() + 0.587 * c1.green() + 0.114 * c1.blue()
        u1 = -0.147 * c1.red() - 0.289 * c1.green() + 0.436 * c1.blue()
        v1 =  0.615 * c1.red() - 0.515 * c1.green() - 0.100 * c1.blue()
        y2 =  0.299 * c2.red() + 0.587 * c2.green() + 0.114 * c2.blue()
        u2 = -0.147 * c2.red() - 0.289 * c2.green() + 0.436 * c2.blue()
        v2 =  0.615 * c2.red() - 0.515 * c2.green() - 0.100 * c2.blue()
        # distance of 2 points in 3D area (YUV cube)
        # the U and V distances are enlarged, because they are harder to see for humans
        distance = math.sqrt((y1-y2)**2 + 3*(u1-u2)**2 + 3*(v1-v2)**2)
        return distance


class TempGui(QtGui.QWidget):
    def __init__(self, num_lines, screen_width):
        super(TempGui, self).__init__()

        self.const = num_lines
        self.cm = ColorManager(screen_width, self.const, "newrand")

        self.setLayout(QtGui.QVBoxLayout())
        widget = QtGui.QWidget()
        widget.setLayout(QtGui.QHBoxLayout())
        self.layout().setAlignment(QtCore.Qt.Alignment(QtCore.Qt.AlignBottom))
        self.text_field = QtGui.QLineEdit()
        self.button = QtGui.QPushButton("Confirm")
        self.button.clicked.connect(self.command)
        widget.layout().addWidget(self.text_field)
        widget.layout().addWidget(self.button)
        self.layout().addWidget(widget)

        self.setAutoFillBackground(True)

        p = self.palette()
        p.setColor(self.backgroundRole(), QtCore.Qt.black)
        self.setPalette(p)

        tracks = []
        for i in range (0, self.const):
            start = random.randint(0, screen_width)
            stop = random.randint(0, screen_width)
            while stop >= start:
                start = random.randint(0, screen_width)
                stop = random.randint(0, screen_width)
            color = QtGui.QColor(0, 0, 0)
            track = Track(start, stop, 0, color)
            tracks.append(track)

        tracks.sort(key=lambda x: x.len)

        for track in tracks:
            self.cm.new_track(track.start, track.stop)

        #for i in range(0, self.const):
        #   self.tracks[i].set_color(self.colors[i])

    def command(self):
        command = self.text_field.text().split(" ")
        self.cm.merge(int(command[0]), int(command[1]))
        self.update()

    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        pen = QtGui.QPen()
        pen.setWidth(10)
        for i in range(0, len(self.cm.tracks)):
            track = self.cm.tracks[i]
            pen.setColor(track.get_color())
            qp.setPen(pen)
            qp.drawLine(track.start, 12*i + 50, track.stop, 12*i + 50)
        qp.end()


class ColorComparatorGui(QtGui.QWidget):
    def __init__(self, limit, screen_width):
        super(ColorComparatorGui, self).__init__()

        self.const = limit
        self.cm = ColorManager(screen_width, self.const, "j")

        self.setAutoFillBackground(True)

        p = self.palette()
        p.setColor(self.backgroundRole(), QtCore.Qt.black)
        self.setPalette(p)

        self.tracks = []
        self.tracks.append(Track(0, screen_width, -1, QtGui.QColor().fromRgb(146, 51, 210)))
        self.tracks.append(Track(0, screen_width, -1, QtGui.QColor().fromRgb(0, 0, 0)))

        for i in range(0, limit):
            self.tracks.append(self.cm.test_dif())

    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        pen = QtGui.QPen()
        pen.setWidth(10)
        for i in range(0, len(self.tracks)):
            track = self.tracks[i]
            pen.setColor(track.get_color())
            qp.setPen(pen)
            qp.drawLine(track.start, 12*i + 50, track.stop, 12*i + 50)
        qp.end()


class Track():
    def __init__(self, start, stop, id, color):
        self.color = color
        if start > stop:
            self.stop = start
            self.start = stop
        else:
            self.start = start
            self.stop = stop
        self.id = id
        self.len = stop - start

    def get_color(self):
        return self.color

    def set_color(self, color):
        self.color = color

    def get_len(self):
        return self.stop - self.start

    def set_color_id(self, color_id):
        self.color_id = color_id


def colorize_project(project):
    from utils.video_manager import get_auto_video_manager
    vid = get_auto_video_manager(project)

    limit = 0
    for _, ch in project.chm.chunks_.iteritems():
        if ch.length() > 0:
            limit += 1

    print limit, "vs. ", len(project.chm.chunks_)

    limit = min(limit, 50)

    project.color_manager = ColorManager(vid.total_frame_count(), limit)
    for ch in project.chm.chunk_list():
        if ch.length() > 0:
            if ch.length() < 100:
                r = random.randint(50, 255)
                g = random.randint(50, 255)
                b = random.randint(50, 255)

                c = QtGui.QColor.fromRgb(r, g, b)
                ch.color = c
            else:
                rch = RegionChunk(ch, project.gm, project.rm)
                ch.color, _ = project.color_manager.new_track(rch.start_frame(), rch.end_frame())

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)

    ex = TempGui(20, 800)
    ex.show()
    ex.move(-500, -500)
    ex.showMaximized()
    ex.setFocus()


    app.exec_()
    for key, track_dict in ex.cm.adjacency.iteritems():
        track_ids = ""
        for track in track_dict:
            track_ids += str(track.id)
            track_ids += " "
        print "%s: %s" % (key, track_ids)
    app.deleteLater()
    sys.exit()