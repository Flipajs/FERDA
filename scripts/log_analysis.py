__author__ = 'flipajs'

from core.project.project import Project
from core.log import ActionNames, LogCategories, Log
from math import copysign
from PyQt4 import QtGui, QtCore
import sys
from gui.correction.case_widget import CaseWidget
from gui.correction.configurations_visualizer import ConfigurationsVisualizer
from utils.video_manager import get_auto_video_manager


def draw_j_mark(node, ex, i):
    mark_size = 10
    ex.scene.addRect(ex.left_margin + ex.w_ * i,
                     ex.top_margin + ex.node_positions[node] * ex.h_,
                     mark_size,
                     mark_size,
                     QtGui.QColor(0, 0, 0, 230),
                     QtGui.QColor(255, 0, 0, 230))

def draw_s_mark(node, ex, i):
    mark_size = 10
    ex.scene.addRect(ex.left_margin + ex.w_ * i + ex.node_size - mark_size,
                     ex.top_margin + ex.node_positions[node] * ex.h_,
                     mark_size,
                     mark_size,
                     QtGui.QColor(0, 0, 0, 230),
                     QtGui.QColor(255, 0, 0, 230))

def test_join(node, solver):
    if solver.g.in_degree(node) > 1:
        for n_in, _ in solver.g.in_edges(node):
            a = (node.area() - n_in.area()) / float(n_in.area())
            if a > 0.7:
                return True

    return False

def test_split(node, solver):
    if solver.g.out_degree(node) > 1:
        for _, n_out in solver.g.out_edges(node):
            a = (node.area() - n_out.area()) / float(n_out.area())
            if a > 0.7:
                return True

    return False

def experiment1(project):
    solver = project.saved_progress['solver']
    g = solver.g
    vid = get_auto_video_manager(project)
    nodes_groups = []
    for i in range(75, 85):
        if i in solver.nodes_in_t:
            nodes_groups.append(solver.nodes_in_t[i])
        else:
            nodes_groups.append([])

    cv = ConfigurationsVisualizer(solver, vid)
    ex = CaseWidget(g, project, nodes_groups, {}, vid, cv)

    for i in range(len(nodes_groups)):
        for n in nodes_groups[i]:
            if test_join(n, solver):
                draw_j_mark(n, ex, i)
            if test_split(n, solver):
                draw_s_mark(n, ex, i)

    # draw_s_mark(test_node, ex, 0)

    ex.show()
    ex.move(-500, -500)
    ex.showMaximized()
    ex.setFocus()

    app.exec_()


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    p = Project()
    # p.load('/Users/flipajs/Documents/wd/ms_colormarks/colormarks.fproj')
    # p.load('/Users/flipajs/Documents/wd/ms_colormarks_mcheck/colormarks.fproj')
    # p.load('/Users/flipajs/Documents/wd/ms_colormarks_mcheck_c10/colormarks.fproj')
    # p.load('/Users/flipajs/Documents/wd/ms_colonyvid5m/colonyvid5m.fproj')
    p.load('/Users/flipajs/Documents/wd/ms_colonyvid5m_mcheck_c10/colonyvid5m.fproj')

    print "# NODES: ", len(p.saved_progress['solver'].g), p.solver_parameters.certainty_threshold
    solver = p.saved_progress['solver']
    # solver.simplify()
    # solver.simplify_to_chunks()
    # solver.save()
    # print "# NODES: ", len(solver.g), p.solver_parameters.certainty_threshold
    # p.load('/Users/flipajs/Documents/wd/c2__/c2.fproj')
    # p.working_directory = '/Users/flipajs/Documents/wd/ms_colonyvid5m_mcheck_c10'
    # p.video_paths = ['/Volumes/Seagate Expansion Drive 1/IST - videos/colonies/F1C5.avi']
    # p.solver_parameters.certainty_threshold = 0.1
    # p.save()

    nodes = []
    for t in range(76, 83):
        # for n in solver.nodes_in_t[t]:
        nodes.append(list(solver.nodes_in_t[t]))

    # nodes = sorted(nodes, key=lambda x: x.frame_)

    for n in reversed(nodes):
        print solver.simplify(n, first_run=True)
        solver.simplify_to_chunks()
        p.saved_progress['solver'] = solver
        experiment1(p)

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

                print "AREA: %d, in_d: %d, out_d: %d, type: %s" % (
                n.area(), solver.g.out_degree(n), solver.g.in_degree(n), l.action_name)
                print "\tIN:"
                for n_in, _, d in solver.g.in_edges(n, data=True):
                    c = 0.00
                    if 'certainty' in d:
                        c = round(d['certainty'], 2)
                    s = -round(d['score'], 2)

                    a = round((n.area() - n_in.area()) / float(n_in.area()), 2)
                    a = str(copysign(1, a) + a) + 'x'

                    print "\t%0.2f \t%0.2f\t %s" % (s, c, a)

                print "\tOUT:"
                for _, n_out, d in solver.g.out_edges(n, data=True):
                    c = 0
                    if 'certainty' in d:
                        c = round(d['certainty'], 2)
                    s = -round(d['score'], 2)

                    a = round((n_out.area() - n.area()) / float(n.area()), 2)
                    a = str(copysign(1, a) + a) + 'x'
                    print "\t%0.2f \t%0.2f\t %s" % (s, c, a)

                if l.action_name == ActionNames.MARK_JOIN:
                    join_num += 1
                elif l.action_name == ActionNames.MARK_SPLIT:
                    split_num += 1
                elif l.action_name == ActionNames.MARK_JOIN_AND_SPLIT:
                    both_num += 1

    print "#JOIN: %d, #SPLIT: %d, #BOTH: %d" % (join_num, split_num, both_num)

    app.deleteLater()
    sys.exit()