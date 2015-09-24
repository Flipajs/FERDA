__author__ = 'flipajs'

from core.project.project import Project
from core.log import ActionNames, LogCategories, Log


if __name__ == "__main__":
    p = Project()
    p.load('/Users/flipajs/Documents/wd/ms_colormarks/colormarks.fproj')
    p.working_directory = '/Users/flipajs/Documents/wd/ms_colormarks/'
    p.video_paths = ['/Users/flipajs/Documents/wd/C210min.avi']

    solver = p.saved_progress['solver']
    join_num = 0
    split_num = 0
    both_num = 0
    for l in p.log.data_:
        if l.category == LogCategories.USER_ACTION:
            if l.action_name in [ActionNames.MARK_JOIN, ActionNames.MARK_SPLIT, ActionNames.MARK_JOIN_AND_SPLIT]:
                n = l.data['node']
                if not n:
                    continue

                print "AREA: %d, in_d: %d, out_d: %d, type: %s" %(n.area(), solver.g.out_degree(n), solver.g.in_degree(n), l.action_name)
                print "\tIN:"
                for n_in, _, d in solver.g.in_edges(n, data=True):
                    c = 0.00
                    if 'certainty' in d:
                        c = round(d['certainty'], 2)
                    s = -round(d['score'], 2)

                    a = round((n.area()-n_in.area())/float(n_in.area()), 2)
                    print "\t", s, "\t", c, "\t", a

                print "\tOUT:"
                for _, n_out, d in solver.g.out_edges(n, data=True):
                    c = 0
                    if 'certainty' in d:
                        c = round(d['certainty'], 2)
                    s = -round(d['score'], 2)

                    a = round((n_out.area()-n.area())/float(n.area()), 2)
                    print "\t", s, "\t", c, "\t", a

                if l.action_name == ActionNames.MARK_JOIN:
                    join_num += 1
                elif l.action_name == ActionNames.MARK_SPLIT:
                    split_num += 1
                elif l.action_name == ActionNames.MARK_JOIN_AND_SPLIT:
                    both_num += 1

    print "#JOIN: %d, #SPLIT: %d, #BOTH: %d" % (join_num, split_num, both_num)

