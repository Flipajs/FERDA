l__author__ = 'flipajs'

from core.project.project import Project


p = Project()

# p.load('/home/simon/FERDA/projects/clusters_gt/Cam1_/cam1.fproj')
# p.video_paths = ['/home/simon/FERDA/projects/clusters_gt/Cam1_/Cam1_clip.avi']
# p.working_directory = '/home/simon/FERDA/projects/clusters_gt/Cam1_/'

p.load('/home/simon/FERDA/projects/clusters_gt/zebrafish/zebrafish.fproj')
p.video_paths = ['/home/simon/FERDA/projects/clusters_gt/zebrafish/5Zebrafish_nocover_22min.avi']
p.working_directory = '/home/simon/FERDA/projects/clusters_gt/zebrafish/'

for it in p.log.data_:
    print it.action_name, it.data

p.save()
