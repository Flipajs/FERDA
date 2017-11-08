from core.project.project import Project

WD = '/Volumes/Seagate Expansion Drive/HH1_PRE_upper_thr/HH1_PRE_upper_thr.fproj'


p = Project()
p.load(WD)

print "TRACKLETS...."
for f in range(7400, 7420):
    print "### ", f
    print "tracklets"
    for r in p.gm.regions_in_t(f):
        print r.id()

    print
    print "regions"
    for t in p.chm.chunks_in_frame(f):
        print t.id(), t.segmentation_class


    print
    print

