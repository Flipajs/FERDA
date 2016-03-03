import sys
from fitting import Fitting
from core.region.region import Region
import cPickle as pickle
from PyQt4 import QtCore
import os

baseDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(baseDir)

s_id = sys.argv[1]
file_path = sys.argv[2]

with open(file_path, 'rb') as f:
    data = pickle.load(f)

f = data['fitting']
results = f.fit()

with open(file_path, 'wb') as f_:
    pickle.dump({'results': results, 'fitting': f}, f_, -1)
