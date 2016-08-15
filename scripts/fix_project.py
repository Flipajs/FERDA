l__author__ = 'flipajs'

from core.project.project import Project


p = Project()

#p.load('/Users/flipajs/Documents/wd/eight_barbara_b/eight.fproj')
p.load('/home/casillas/Documents/FERDAWD/test3/test.fproj')

p.video_paths = ['/home/casillas/Documents/FERDAWD/eight.m4v']
p.working_directory = '/home/casillas/Documents/FERDAWD/'

for it in p.log.data_:
    print it.action_name, it.data

p.save()
