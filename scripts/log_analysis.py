__author__ = 'flipajs'

from core.project.project import Project
from core.log import ActionNames, LogCategories


if __name__ == "__main__":
    p = Project()
    p.load('/Users/flipajs/Documents/wd/ms_colormarks/colormarks.fproj')
    p.working_directory = '/Users/flipajs/Documents/wd/ms_colormarks/'
    p.video_paths = ['/Users/flipajs/Documents/wd/C210min.avi']


    join_num = 0
    split_num = 0
    both_num = 0
    for l in p.log.data_:
        if l.category == LogCategories.USER_ACTION:
            if l.action_name == ActionNames.