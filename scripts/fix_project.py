__author__ = 'flipajs'

from core.project.project import Project


p = Project()
p.load('/home/ferda/PROJECTS/eight_22/eight22.fproj')

p.video_paths = ['/home/ferda/PROJECTS/eight.m4v']
p.working_directory = '/home/ferda/FERDA'

for it in p.log.data_:
    print it.action_name, it.data

p.save()