import numpy as np
from matplotlib import pyplot as plt


class clicker_class(object):
    """
    credit: tcaswell on http://stackoverflow.com/questions/19592422/python-gui-that-draw-a-dot-when-clicking-on-plot
    """
    def __init__(self, ax, pix_err=1):
        self.canvas = ax.get_figure().canvas
        self.cid = None
        self.pt_lst = []
        self.pt_plot = ax.plot([], [], marker='+', mec='k', ms=20)[0]
        self.pix_err = pix_err
        self.connect_sf()

        self.__objs = []
        self.ants_pts = []
        self.current_ant = 0
        self.part_order = ['left antenna', 'right antenna', '1st right', '2nd rigth', '3rd right', '3rd left', '2nd left', '1st left', 'DONE']

    def set_visible(self, visible):
        '''sets if the curves are visible '''
        self.pt_plot.set_visible(visible)

    def clear(self):
        '''Clears the points'''
        self.pt_lst = []
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
            self.pt_lst.append((event.xdata, event.ydata))
        elif event.button == 3:
            self.remove_pt((event.xdata, event.ydata))

        self.redraw()

    def key_press_event(self, event):
        if event.key == 'ctrl+C':
            self.clear()
            return

    def remove_pt(self, loc):
        if len(self.pt_lst) > 0:
            id_ = np.argmin(map(lambda x: np.sqrt((x[0] - loc[0]) ** 2 + (x[1] - loc[1]) ** 2), self.pt_lst))
            if np.linalg.norm(np.array(loc) - np.array(self.pt_lst[id_])) < 30:
                self.pt_lst.pop(id_)

    def redraw(self):
        for o in self.__objs:
            o.remove()

        self.__objs = []

        plt.hold(True)
        i = 0
        for x, y in self.pt_lst:
            c = 'r'

            o = plt.scatter(x, y, c=c, marker='o', s=70, edgecolors='k', alpha=0.4)
            self.__objs.append(o)

            # sig1 = -1
            # sig2 = 1
            #
            # off = 20
            # o = ax.annotate(i, xy=(x, y), xycoords='data',
            #     xytext=(sig1*off, sig2*off), textcoords='offset points',
            #             bbox=dict(boxstyle="round", fc="red", alpha=0.5),
            #     arrowprops=dict(arrowstyle="->"),
            #     )

            # self.__objs.append(o)

            i += 1


        o = ax.annotate('Ant #'+str(self.current_ant+1), xy=(0, -15),
                bbox=dict(boxstyle="round", fc="green", alpha=0.7),
                )
        self.__objs.append(o)

        t = 'ERROR'
        if len(self.pt_lst) < len(self.part_order):
            t = self.part_order[len(self.pt_lst)]

        o = ax.annotate(t, xy=(80, -15),
                bbox=dict(boxstyle="round", fc="green", alpha=0.7),
                )
        self.__objs.append(o)

        plt.hold(False)

        self.canvas.draw()

    def return_points(self):
        '''Returns the clicked points in the format the rest of the
        code expects'''
        return np.vstack(self.pt_lst).T


"""

ctrl + shift + c - clear all
s - skip
1 2 3 .... pt number

"""


# im = plt.imread('/Users/flipajs/Desktop/Screen Shot 2016-03-01 at 17.28.47.png')
im = plt.imread('/Users/flipajs/Desktop/Screen Shot 2016-03-09 at 08.14.06.png')
fig = plt.figure()
ax = fig.add_subplot(111)
plt.imshow(im)
# ax = plt.gca()
cc = clicker_class(ax)
plt.show()