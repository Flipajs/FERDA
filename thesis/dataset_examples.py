from __future__ import absolute_import
from .thesis_utils import load_all_projects, project_paths
import matplotlib.pyplot as plt
import cv2
from utils.video_manager import get_auto_video_manager
from thesis.config import *
import matplotlib.gridspec as gridspec


if __name__ == '__main__':
    projects = load_all_projects()

    frames = {'Cam1': [986, 1306, 1696, 2237, 2451, 2717],
              'Zebrafish': [986, 1306, 1696, 2237, 2451, 2717],
              'Camera3': [986, 1306, 1696, 2237, 2451, 2717],
              'Sowbug3': [986, 1306, 1696, 2237, 2451, 2717],
              }

    crops = {'Cam1': None,
             'Zebrafish': None,
             'Camera3': None,
             'Sowbug3': None
             }


    for p_name, p in projects.iteritems():
        # if p_name != 'Camera3':
        #     continue

        vm = get_auto_video_manager(p)
        imgs = []

        i = 0
        for frame in frames[p_name]:
            i += 1
            plt.subplot(2, 3, i)
            im = vm.get_frame(frame).copy()
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            imgs.append(im)
            plt.imshow(imgs[-1])
            plt.axis('off')
            plt.title('frame: '+str(frame), fontsize=7)

        plt.subplots_adjust(wspace=0, hspace=0)

        plt.savefig(OUT_IMGS_WD+'/dataset/'+p_name+'.png', bbox_inches='tight', pad_inches=0, dpi=256)
        # plt.show()


