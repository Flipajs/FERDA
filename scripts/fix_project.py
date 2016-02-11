__author__ = 'flipajs'

from core.project.project import Project


p = Project()

p.load('/home/sheemon/FERDA/projects/dita_proj/dita.fproj')
p.video_paths = ['/home/sheemon/FERDA/projects/c5_0h03m-0h06m.avi']
p.working_directory = '/home/sheemon/FERDA/projects/dita_proj'

for it in p.log.data_:
    print it.action_name, it.data

p.save()