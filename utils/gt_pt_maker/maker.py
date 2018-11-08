from __future__ import print_function
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import object
import numpy as np
from matplotlib import pyplot as plt
import pickle as pickle
import cv2

class clicker_class(object):
    """
    credit: tcaswell on http://stackoverflow.com/questions/19592422/python-gui-that-draw-a-dot-when-clicking-on-plot
    """
    def __init__(self, ax, vm, out_path, set_id=0, frame=0, del_limit=30):
        self.canvas = ax.get_figure().canvas
        self.cid = None
        # form data[frame][antid]
        self.frame = frame
        self.data = {frame: {0: []}}
        self.pt_plot = ax.plot([], [], marker='+', mec='k', ms=20)[0]
        self.del_limit = del_limit
        self.connect_sf()

        self.__objs = []
        self.ants_pts = []
        self.current_ant = 0
        self.part_order = ['left antenna', 'right antenna', '1st right', '2nd rigth', '3rd right', '3rd left', '2nd left', '1st left', 'DONE']

        self.set_id = set_id
        self.vm = vm
        self.out_path = out_path
        self.set_frame()

    def set_frame(self):
        self.data.setdefault(self.frame, {}).setdefault(0, [])
        self.__objs = []
        self.current_ant = 0
        plt.cla()
        im = cv2.cvtColor(vm.get_frame(self.frame), cv2.COLOR_BGR2RGB)
        plt.imshow(im)
        self.redraw()

    def set_visible(self, visible):
        '''sets if the curves are visible '''
        self.pt_plot.set_visible(visible)

    def clear(self):
        '''Clears the points'''
        self.data[self.frame] = {}
        self.redraw()

    def connect_sf(self):
        if self.cid is None:
            self.cid = self.canvas.mpl_connect('button_press_event',
                                               self.click_event)
            self.cid2 = self.canvas.mpl_connect('key_press_event',
                                               self.key_press_event)

    def disconnect_sf(self):
        if self.cid is not None:
            self.canvas.mpl_disconnect(self.cid)
            self.cid = None

    def click_event(self, event):
        ''' Extracts locations from the user'''
        if event.xdata is None or event.ydata is None:
            return
        if event.button == 1:
            if event.key == 'z':
                x, y = event.xdata, event.ydata
                offset = 75
                plt.xlim([x-offset, x+offset])
                plt.ylim([y-offset, y+offset])
                plt.gca().invert_yaxis()

            elif event.key == 'Z':
                ax.autoscale()
            else:
                self.data[self.frame][self.current_ant].append((event.xdata, event.ydata))
        elif event.button == 3:
            self.remove_pt((event.xdata, event.ydata))

        self.redraw()

    def key_press_event(self, event):
        if event.key == 'ctrl+C':
            self.clear()
            return

        elif event.key == 'n':
            self.current_ant += 1
            self.data[self.frame].setdefault(self.current_ant, [])

        elif event.key == 'b':
            if self.current_ant > 0:
                self.current_ant -= 1

        elif event.key == 'g':
            if self.frame > 0:
                self.frame -= 1
            self.set_frame()
            self.save()
            return

        elif event.key == 'h':
            self.frame += 1
            self.set_frame()
            self.save()
            return

        elif event.key == 'S':
            self.save()

        self.redraw()

    def save(self):
        import time
        timestr = time.strftime("%Y%m%d-%H%M%S")
        with open(self.out_path+'/'+str(self.set_id)+'_'+timestr+'.pkl', 'wb') as f:
            p = pickle.Pickler(f, -1)
            p.dump(self.data)

        print('SAVED')


    def remove_pt(self, loc):
        if len(self.data[self.frame][self.current_ant]) > 0:
            id_ = np.argmin([np.sqrt((x[0] - loc[0]) ** 2 + (x[1] - loc[1]) ** 2) for x in self.data[self.frame][self.current_ant]])
            if np.linalg.norm(np.array(loc) - np.array(self.data[self.frame][self.current_ant][id_])) < self.del_limit:
                self.data[self.frame][self.current_ant].pop(id_)

    def redraw(self):
        for o in self.__objs:
            o.remove()

        self.__objs = []

        plt.hold(True)
        colors = ['r', 'g', 'b', 'm', 'c', 'y']

        for a_id in self.data[self.frame]:
            i = 0
            for x, y in self.data[self.frame][a_id]:
                m = 'x' if i < 2 else '+'
                o = plt.scatter(x, y, c=colors[a_id%len(colors)], marker=m, s=100, edgecolors='k', alpha=0.7)
                self.__objs.append(o)

                i += 1

        s1 = 'Frame: '+str(self.frame)+' Ant #'+str(self.current_ant+1)

        s2 = 'ERROR'
        if len(self.data[self.frame][self.current_ant]) < len(self.part_order):
            s2 = self.part_order[len(self.data[self.frame][self.current_ant])]

        plt.title(s1+', '+s2)
        plt.hold(False)

        self.canvas.draw()

    def return_points(self):
        '''Returns the clicked points in the format the rest of the
        code expects'''
        # return np.vstack(self.pt_lst).T
        return None


"""
n - next ant
b - previous ant
shift+s - save
z + click - zoom in
shift + z + click - zoom out

right-click - remove point

"""

data = None
with open('/Users/flipajs/Documents/wd/antennas_gt/Cam1/50.pkl', 'rb') as f:
    up = pickle.Unpickler(f)
    data = up.load()

fig = plt.figure()
ax = fig.add_subplot(111)
from utils.video_manager import VideoManager
vm = VideoManager('/Users/flipajs/Documents/wd/Cam1_clip.avi')
cc = clicker_class(ax, vm, '/Users/flipajs/Documents/wd/antennas_gt/test', set_id=50, frame=50)

if data is not None:
    cc.data = data
    cc.frame = sorted([f for f in data])[0]
    cc.set_frame()
plt.show()
