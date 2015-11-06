__author__ = 'flipajs'

from core.project.project import Project


p = Project()
p.load('/home/sheemon/FERDA/projects/eight_new/eight.fproj')

p.video_paths = ['/home/sheemon/FERDA/projects/eight.m4v']
p.working_directory = '/home/sheemon/FERDA/projects/eight_new'

for it in p.log.data_:
    print it.action_name, it.data

p.save()