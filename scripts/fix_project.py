l__author__ = 'flipajs'

from core.project.project import Project


p = Project()

p.load('../projects/Sowbug3-fixed-segmentation/Sowbug-fixed-segmentation.fproj', lightweight=True)
p.video_paths = ['/datagrid/ferda/data/youtube/Sowbug3.mp4']
# p.working_directory = '/home/simon/FERDA/projects/Cam1_/'

for it in p.log.data_:
    print it.action_name, it.data

p.save()
