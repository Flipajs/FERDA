from core.project.project import Project

WD = '/Users/flipajs/Documents/wd/FERDA/'
DEV_WD = '/Users/flipajs/Documents/dev/ferda'
OUT_WD = DEV_WD+'/thesis/out/'
OUT_IMGS_WD = OUT_WD+'/imgs/'
RESULT_WD = DEV_WD+'/thesis/results'
RESULTS_WD = RESULT_WD

cam1_path = WD+'Cam1_playground'
sowbug3_path = WD+'Sowbug3'
zebrafish_path = WD+'zebrafish_playground'
camera3_path = WD+'Camera3'

project_paths = {'Cam1': cam1_path, 'Sowbug3': sowbug3_path, 'Zebrafish': zebrafish_path, 'Camera3': camera3_path}
project_colors = {'Cam1': 'r', 'Sowbug3': 'g', 'Zebrafish': 'b', 'Camera3': 'm'}
project_real_names = {'Cam1': 'Ants-1', 'Camera3': 'Ants-3', 'Zebrafish': 'Zebrafish-1', 'Sowbug3': 'Sowbug-3'}
project_marks = {'Cam1': 'o', 'Sowbug3': 'v', 'Zebrafish': 's', 'Camera3': '*'}


idTracker_results_paths = {'Cam1': '/Users/flipajs/Documents/wd/idTracker/Cam1/trajectories',
                     'Sowbug3': '/Users/flipajs/Documents/wd/idTracker/Sowbug3/trajectories',
                     'Zebrafish': '/Users/flipajs/Documents/wd/idTracker/zebrafish/trajectories',
                     'Camera3': '/Users/flipajs/Documents/wd/idTracker/Camera3/trajectories'}