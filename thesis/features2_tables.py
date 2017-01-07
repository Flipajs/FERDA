from thesis.config import *
import cPickle as pickle
import numpy as np
from pylatex.utils import NoEscape

FORMAT_PERCENTS = "{:.2%}"

def is_positive(x):
    f = FORMAT_PERCENTS
    s = f.format(x)
    if x > 0:
        return NoEscape(r'\cellcolor{LimeGreen}') + NoEscape(r'\textbf{' + '+{:.2f}\%'.format(x * 100) + '}')
    else:
        return s

def best(val, all, func=max):
    f = FORMAT_PERCENTS

    s = f.format(val)
    if func(all) == val:
        return NoEscape(r'\cellcolor{LimeGreen}')+NoEscape(r'\textbf{'+'{:.2f}\%'.format(val*100)+'}')
    else:
        return s


def comparison2latex(name1='default', name2='', out_name='', highlight=True):
    from pylatex import Document, Section, Subsection, Tabular, Tabularx, MultiColumn, MultiRow
    from pylatex.utils import bold, italic, verbatim, escape_latex, NoEscape
    from pylatex.package import Package

    comparison = True
    if name2 is None:
        comparison = False

    with open(RESULTS_WD+'/features_'+name1) as f:
        results1 = pickle.load(f)

    if comparison:
        with open(RESULTS_WD+'/features_'+name2) as f:
            results2 = pickle.load(f)


    doc = Document("multirow")
    doc.packages.add(Package('xcolor', options='table, dvipsnames'))
    # doc.append(Package('xcolors', options='table'))
    keys = ['Cam1', 'Zebrafish', 'Camera3', 'Sowbug3']

    # table1 = Tabular('|c|c|c|c|', booktabs=True)
    table1 = Tabular('|r||c|c|c|c|')
    table1.add_hline()
    table1.add_row('',
        bold(project_real_names['Cam1']),
        bold(project_real_names['Zebrafish']),
        bold(project_real_names['Camera3']),
        bold(project_real_names['Sowbug3']))

    table1.add_hline()
    table1.add_hline()
    ts = 0.95

    features = ['basic', 'colornames', 'idtracker_c', 'idtracker_i', 'lbp', 'hog']
    features_names = {'basic': 'Moments', 'colornames': 'Colornames', 'idtracker_c': 'C-co-occurence', 'idtracker_i': 'I-co-occurence', 'lbp': 'LBP', 'hog': 'HoG'}
    vvals = []
    for feat in features:
        vals = []
        for key in keys:
            r1_acc = results1[0][key][ts]['fm_'+feat+' ']['accuracy']
            x1 = np.mean(r1_acc)
            if comparison:
                r2_acc = results2[0][key][ts]['fm_'+feat+' ']['accuracy']
                x2 = np.mean(r2_acc)

                vals.append(x2 - x1)
            else:
                vals.append(x1)

        if comparison and highlight:
            table1.add_row(bold(features_names[feat]),
                           is_positive(vals[0]),
                           is_positive(vals[1]),
                           is_positive(vals[2]),
                           is_positive(vals[3]))
        else:
            f = FORMAT_PERCENTS
            table1.add_row(bold(features_names[feat]), f.format(vals[0]), f.format(vals[1]),
                           f.format(vals[2]), f.format(vals[3]))

        table1.add_hline()

        vvals.append(vals)

    vvals = np.array(vvals)
    if comparison and highlight:
        table1.add_hline()
        vals = np.mean(vvals, axis=0)
        table1.add_row(bold('An average'),
                       is_positive(vals[0]),
                       is_positive(vals[1]),
                       is_positive(vals[2]),
                       is_positive(vals[3]))

        table1.add_hline()


    # f = FORMAT_PERCENTS
    # for key in keys:
    #     x = results[key]
    #     xx = results[key+'_nogaps']
    #     table1.add_hline()
    #
    #     all = [x[2], x[0], xx[0]]
    #     table1.add_row((MultiRow(3, data=bold(key)), italic('correct'), best(x[2], all), best(x[0], all), best(xx[0], all)))
    #     table1.add_hline(start=2)
    #     all = [x[3], x[1], xx[1]]
    #     table1.add_row('', italic('wrong'), best(x[3], all, min), best(x[1], all, min), best(xx[1], all, min))
    #     table1.add_hline(start=2)
    #     all = [1-x[2]-x[3], 1-x[0]-x[1], 1-xx[0]-xx[1]]
    #     table1.add_row('', italic('unassigned'), f.format(all[0]), f.format(all[1]), f.format(all[2]))
    #     table1.add_hline()

    doc.append(table1)
    if not len(out_name):
        out_name = name2

    doc.generate_pdf(OUT_WD+'/tables/features_comparison_'+out_name)
    table1.generate_tex(OUT_WD+'/tables/features_comparison_'+out_name)



def overall2latex(name1='', name2='', out_name='', highlight=True, nnval=None):
    from pylatex import Document, Section, Subsection, Tabular, Tabularx, MultiColumn, MultiRow
    from pylatex.utils import bold, italic, verbatim, escape_latex, NoEscape
    from pylatex.package import Package

    comparison = True
    if name2 is None:
        comparison = False

    with open(RESULTS_WD+'/features_'+name1) as f:
        results1 = pickle.load(f)

    if comparison:
        with open(RESULTS_WD+'/features_'+name2) as f:
            results2 = pickle.load(f)

    doc = Document("multirow")
    doc.packages.add(Package('xcolor', options='table, dvipsnames'))
    # doc.append(Package('xcolors', options='table'))
    keys = ['Cam1', 'Zebrafish', 'Camera3', 'Sowbug3']

    # table1 = Tabular('|c|c|c|c|', booktabs=True)
    table1 = Tabular('|r||c|c|c|c|')
    table1.add_hline()
    table1.add_row('',
        bold(project_real_names['Cam1']),
        bold(project_real_names['Zebrafish']),
        bold(project_real_names['Camera3']),
        bold(project_real_names['Sowbug3']))

    table1.add_hline()
    table1.add_hline()
    ts = 0.95

    vals1 = []
    vals2 = []
    #
    # # computed, but not saved..
    # vals3 = [0, np.mean([0.8398, 0.8437, 0.8475]), 0, 0]

    vals3 = nnval

    for key in keys:
        r1_acc = results1[0][key][ts]['all']['accuracy']
        x1 = np.mean(r1_acc)

        r2_acc = results2[0][key][ts]['all']['accuracy']
        x2 = np.mean(r2_acc)

        vals1.append(x1)
        vals2.append(x2)

    f = FORMAT_PERCENTS
    table1.add_row(bold('RFC default'), best(vals1[0], [vals1[0], vals2[0], vals3[0]]), best(vals1[1], [vals1[1], vals2[1], vals3[1]]),
                   best(vals1[2], [vals1[2], vals2[2], vals3[2]]), best(vals1[3], [vals1[3], vals2[3], vals3[3]]))

    table1.add_hline()
    table1.add_row(bold('RFC tuned'), best(vals2[0], [vals1[0], vals2[0], vals3[0]]), best(vals2[1], [vals1[1], vals2[1], vals3[1]]),
                   best(vals2[2], [vals1[2], vals2[2], vals3[2]]), best(vals2[3], [vals1[3], vals2[3], vals3[3]]))

    table1.add_hline()
    table1.add_row(bold('NN'),
                   best(vals3[0], [vals1[0], vals2[0], vals3[0]]),
                   best(vals3[1], [vals1[1], vals2[1], vals3[1]]),
                   best(vals3[2], [vals1[2], vals2[2], vals3[2]]),
                   best(vals3[3], [vals1[3], vals2[3], vals3[3]]))

    table1.add_hline()

    doc.append(table1)
    if not len(out_name):
        out_name = name2

    doc.generate_pdf(OUT_WD+'/tables/features_full_'+out_name)
    table1.generate_tex(OUT_WD+'/tables/features_full_'+out_name)



if __name__ == '__main__':
    #
    # comparison2latex(name2=None, out_name='default')
    # #
    #
    # comparison2latex(name2='gini')
    #
    #
    # comparison2latex(name2='min_samples_leaf_2')
    # comparison2latex(name2='min_samples_leaf_3')
    # comparison2latex(name2='min_samples_leaf_5')
    #
    #
    # comparison2latex(name2='max_depth_5')
    # comparison2latex(name2='max_depth_10')
    # comparison2latex(name2='max_depth_15')
    # comparison2latex(name2='max_depth_20')
    # comparison2latex(name2='max_depth_25')
    # comparison2latex(name2='max_depth_50')
    # comparison2latex(name2='max_depth_100')
    #
    #
    # comparison2latex(name2='max_features_auto')
    # comparison2latex(name2='max_features_10')
    # comparison2latex(name2='max_features_20')
    # comparison2latex(name2='max_features_30')
    # comparison2latex(name2='max_features_40')
    # comparison2latex(name2='max_features_50')
    comparison2latex(name2='max_features_60')
    comparison2latex(name2='max_features_70')
    comparison2latex(name2='max_features_80')
    # comparison2latex(name2='max_features_75')
    # comparison2latex(name2='max_features_100')
    #
    #
    # comparison2latex(name2='n_estimators_20')
    # comparison2latex(name2='n_estimators_30')
    # comparison2latex(name2='n_estimators_40')
    # comparison2latex(name2='n_estimators_50')
    # comparison2latex(name2='n_estimators_75')
    # comparison2latex(name2='n_estimators_100')
    # comparison2latex(name2='n_estimators_200')

    if False:
        nn = []
        for pname in ['Cam1', 'Zebr', 'Came', 'Sowb']:
            out_name = 'results/nn_'+pname+'-c.pkl'
            with open(out_name) as f:
                resultsc = pickle.load(f)['fm_idtracker_c.sqlite3']
                # resultsc = pickle.load(f)

            out_name = 'results/nn_'+pname+'-i.pkl'
            with open(out_name) as f:
                resultsi = pickle.load(f)['fm_idtracker_i.sqlite3']

            vals = []
            for i in range(len(resultsc)):
                r1 = np.array(resultsc[i][0])
                r2 = np.array(resultsi[i][0])
                c1 = np.array(resultsc[i][3])
                c2 = np.array(resultsi[i][3])

                x = np.logical_and(np.logical_and(r1 == r2, c1), c2)
                print "{:.2%}".format(np.sum(x) / float(r1.shape[0]))
                print np.sum(x), r1.shape[0], resultsc[i][1], resultsi[i][1]

                vals.append(np.sum(x) / float(r1.shape[0]))

            nn.append(np.mean(vals))

        overall2latex('default_all', 'best1_all', 'best1', nnval=nn)

    pass