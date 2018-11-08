from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
import pyximport; pyximport.install()

from . import features2
from .feature_manager import FeatureManager
from utils.misc import print_progress

if __name__ == '__main__':
    from core.project.project import Project
    import pickle as pickle

    wd = '/Users/flipajs/Documents/wd/FERDA/Cam1_playground'
    # wd = '/Users/flipajs/Documents/wd/FERDA/zebrafish_playground'
    # wd = '/Users/flipajs/Documents/wd/FERDA/Camera3'
    # wd = '/Users/flipajs/Documents/wd/FERDA/Sowbug3'
    p = Project()
    p.load(wd)

    from core.graph.chunk_manager import ChunkManager

    p.chm = ChunkManager()
    with open(wd + '/temp/isolation_score.pkl', 'rb') as f:
        up = pickle.Unpickler(f)
        p.gm.g = up.load()
        up.load()
        chm = up.load()
        p.chm = chm

    from core.region.region_manager import RegionManager

    p.rm = RegionManager(wd + '/temp', db_name='part0_rm.sqlite3')
    p.gm.rm = p.rm


    with open(p.working_directory+'/temp/test_regions.pkl') as f:
        regions = pickle.load(f)


    import time

    t = time.time()

    for r in regions:
        features2.get_idtracker_features(r, p)

    print(time.time() - t)