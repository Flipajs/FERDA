import sys
import cPickle as pickle
import random
from PyQt4 import QtGui

"""
Use this script in case when color manager was disabled.
Goes through all chunks and set random color.
"""


if __name__ == "__main__":
    working_dir = sys.argv[1]

    with open(working_dir+'/chunk_manager.pkl', 'rb') as f:
        chm = pickle.load(f)

    for ch in chm.chunk_gen():
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        ch.color = QtGui.QColor.fromRgb(r, g, b)

        if not hasattr(ch, 'N'):
            ch.N = set()
            ch.P = set()

    with open(working_dir+'/chunk_manager.pkl', 'wb') as f:
        pickle.dump(chm, f)