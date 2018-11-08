from __future__ import unicode_literals
__author__ = 'flipajs'

import numpy as np


def line_picker(line, me):
    if me.xdata is None: return False, dict()
    x, y = me.x, me.y
    xdata, ydata = line.axes.transData.transform(np.array(line.get_data()).T).T
    index = np.arange(len(xdata))
    index2 = np.linspace(0, index[-1], 2000)
    xdata2 = np.interp(index2, index, xdata)
    ydata2 = np.interp(index2, index, ydata)
    d = np.sqrt((xdata2-x)**2. + (ydata2-y)**2.)
    if np.min(d) < 5:
        return True, {}
    else:
        return False, {}
