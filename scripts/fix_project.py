l__author__ = 'flipajs'

from core.project.project import Project


p = Project()

#p.load('/Users/flipajs/Documents/wd/eight_barbara_b/eight.fproj')
p.load('/home/simon/FERDA/projects/CompleteGraph/CompleteGraph.fproj')

p.video_paths = ['/home/simon/FERDA/projects/CompleteGraph/Cam1_clip.avi']
p.working_directory = '/home/simon/FERDA/projects/CompleteGraph'

for it in p.log.data_:
    print it.action_name, it.data

p.save()
