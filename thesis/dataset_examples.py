from thesis_utils import load_all_projects, project_paths
import matplotlib.pyplot as plt
import cv2
from utils.video_manager import get_auto_video_manager


if __name__ == '__main__':
    projects = load_all_projects()

    frames = {'Cam1': [],
              'Zebrafish': [],
              'Camera3': [986, 1306, 1696, 2237, 2451, 2717],
              'Sowbug3': []
              }

    crops = {'Cam1': None,
             'Zebrafish': None,
             'Camera3': None,
             'Sowbug3': None
             }


    for p_name, p in projects.iteritems():
        if p_name != 'Camera3':
            continue

        vm = get_auto_video_manager(p)
        imgs = []

        i = 0
        for frame in frames[p_name]:
            i += 1
            plt.subplot(3, 2, i)
            im = vm.get_frame(frame).copy()
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            imgs.append(im)
            plt.imshow(imgs[-1])
            plt.axis('off')
            plt.title('frame: '+str(frame))

        plt.savefig('/Users/flipajs/Desktop/pict.png', bbox_inches='tight', pad_inches=0, dpi=256)
        # plt.show()


