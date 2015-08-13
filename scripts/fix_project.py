__author__ = 'flipajs'

from core.project.project import Project


p = Project()
p.load('/Users/flipajs/Documents/wd/eight_barbara_bug/eight.fproj')

p.video_paths = ['/Users/flipajs/Documents/wd/eight.m4v']
p.working_directory = '/Users/flipajs/Documents/wd/eight_barbara_bug'

for it in p.log.data_:
    print it.action_name, it.data

p.save()