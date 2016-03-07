__author__ = 'flipajs'

from core.project.project import Project


p = Project()

p.load('/home/dita/PycharmProjects/FERDA projects/Cam1_orig_/cam1.fproj')
p.video_paths = ['/home/dita/PycharmProjects/FERDA projects/Cam1_/Cam1_clip.avi']
p.working_directory = '/home/dita/PycharmProjects/FERDA projects/Cam1_orig_'

for it in p.log.data_:
    print it.action_name, it.data

p.save()