import os
import sys
baseDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(baseDir)

from .fitting import Fitting
from core.region.region import Region
import pickle as pickle
from PyQt4 import QtCore


s_id = sys.argv[1]
file_path = sys.argv[2]

with open(file_path, 'rb') as f:
    data = pickle.load(f)

f = data['fitting']
results = f.fit()

with open(file_path, 'wb') as f_:
    pickle.dump({'results': results, 'fitting': f}, f_, -1)
