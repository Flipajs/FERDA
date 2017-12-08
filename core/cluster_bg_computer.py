import sys
from core.project.project import Project
from bg_computer_assembling import assembly_after_parallelization
from core.fake_background_computer import FakeBGComp


if __name__ == '__main__':
    working_dir = sys.argv[1]
    part_num = int(sys.argv[2])

    p = Project()
    p.load(working_dir)

    bgcomp = FakeBGComp(p, 0, part_num)
    bgcomp.project = p
    bgcomp.project.is_cluster_ = True
    bgcomp.part_num = part_num

    assembly_after_parallelization(bgcomp)


    print
    print "PROJECT WAS ASSEMBLED."
    print

    print "SAVING..."
    p.save()
    print "SAVED"
