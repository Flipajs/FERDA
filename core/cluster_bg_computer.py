import sys
from core.project.project import Project
from bg_computer_assembling import assembly_after_parallelization

class BGComp():
    def __init__(self):
        self.project = None
        self.part_num = -1

if __name__ == '__main__':
    working_dir = sys.argv[1]
    proj_name = sys.argv[2]
    part_num = sys.argv[3]

    p = Project()
    p.load(working_dir+'/'+proj_name+'.fproj')

    bgcomp = BGComp()
    bgcomp.project = p
    bgcomp.part_num = part_num

    assembly_after_parallelization(bgcomp, cluster=True)

    print
    print "PROJECT WAS ASSEMBLED."
    print