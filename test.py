import sys
from core.project.project import Project
from core.graph_assembly import graph_assembly
from core.fake_background_computer import FakeBGComp


if __name__ == '__main__':
    working_dir = sys.argv[1]
    part_num = int(sys.argv[2])

    p = Project()
    p.load(working_dir)

    bgcomp = FakeBGComp(p, 0, part_num)
    bgcomp.project = p
    bgcomp.part_num = part_num

    graph_assembly(bgcomp)

    print
    print "PROJECT WAS ASSEMBLED."
    print

    print "SAVING..."
    p.save()
    print "SAVED"
