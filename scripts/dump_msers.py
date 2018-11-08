from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import range
from core.project.project import Project
from utils.video_manager import get_auto_video_manager
import pickle as pickle

wd1 = '/Users/flipajs/Documents/wd/FERDA/Cam1_playground'
wd2 = '/Users/flipajs/Documents/wd/FERDA/Camera3'
wd3 = '/Users/flipajs/Documents/wd/FERDA/Zebrafish_playground'

wds = [wd1, wd2, wd3]
for wd in wds:
    p = Project()
    p.load_semistate(wd, 'id_classified_HIL_init_0')

    vm = get_auto_video_manager(p)


    data = {}
    for frame in range(100):
        img = vm.get_frame(frame)

        data[frame] = {'img': img, 'MSERs': []}

        for r in p.gm.regions_in_t(frame):
            rd = {'pts': r.pts(), 'mu20': r.sxx_, 'mu02': r.syy_, 'mu11': r.sxy_, 'centroid': r.centroid()}
            data[frame]['MSERs'].append(rd)

    with open('temp/'+wd.split('/')[-1]+'.pkl', 'wb') as f:
        pickle.dump(data, f)