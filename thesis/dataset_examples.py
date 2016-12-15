from thesis_utils import load_all_projects, project_paths
import matplotlib.pyplot as plt

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

        for i, frame in enumerate(1, len(frames[p_name]) + 1):
            plt.subplot(3, 2, i)
            imgs.append(vm.get_frame(frame).copy())
            plt.imshow(imgs[-1])

        plt.tight_layout()
        plt.show()

