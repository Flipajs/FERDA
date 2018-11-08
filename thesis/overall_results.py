from __future__ import print_function
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from thesis.thesis_utils import load_all_projects
from utils.idtracker import load_idtracker_data
from utils.gt.evaluator import compare_trackers
import pickle as pickle
from thesis.config import *

FORMAT_PERCENTS = "{:.2%}"
def run(semistate='id_classified', dir_name='', HIL=False):
    ps = load_all_projects(semistate=semistate, update_t_nodes=True, add_single_vertices=True)

    results = {}
    for nogaps in ['', '_nogaps']:
        for name in project_paths.keys():
            if name not in ps:
                continue

            print(name)

            p = ps[name]
            path = idTracker_results_paths[name] + nogaps + '.mat'
            impath = DEV_WD + '/thesis/out/imgs/' + dir_name + '/' + name + nogaps + '.png'

            print(path, p.working_directory, impath)
            r = compare_trackers(p, path, impath=impath, name=project_real_names[name])

            results[name + nogaps] = r

    # name = '/thesis/results/overall'
    
    # with open(DEV_WD+ name +'.pkl', 'wb') as f:
    #     pickle.dump(results, f)

def best(val, all, func=max):
    f = FORMAT_PERCENTS

    from pylatex.utils import NoEscape
    s = f.format(val)
    if func(all) == val:
        return NoEscape(r'\cellcolor{LimeGreen}')+NoEscape(r'\textbf{'+'{:.2f}\%'.format(val*100)+'}')
    else:
        return s

def results2latex(name='overall'):
    from pylatex import Document, Section, Subsection, Tabular, Tabularx, MultiColumn, MultiRow
    from pylatex.utils import bold, italic, verbatim, escape_latex, NoEscape
    from pylatex.package import Package
    with open(DEV_WD+ '/thesis/results/overall.pkl') as f:
        results = pickle.load(f)

    doc = Document("multirow")
    doc.packages.add(Package('xcolor', options='table, dvipsnames'))
    # doc.append(Package('xcolors', options='table'))
    keys = ['Cam1', 'Zebrafish', 'Camera3', 'Sowbug3']

    # table1 = Tabular('|c|c|c|c|', booktabs=True)
    table1 = Tabular('|c|c||c|c|c|')
    table1.add_hline()
    table1.add_row('', '', bold('FERDA'), bold('idTracker'), bold('idTracker I'))
    table1.add_hline()

    f = FORMAT_PERCENTS
    for key in keys:
        x = results[key]
        xx = results[key+'_nogaps']
        table1.add_hline()

        all = [x[2], x[0], xx[0]]
        table1.add_row((MultiRow(3, data=bold(key)), italic('correct'), best(x[2], all), best(x[0], all), best(xx[0], all)))
        table1.add_hline(start=2)
        all = [x[3], x[1], xx[1]]
        table1.add_row('', italic('wrong'), best(x[3], all, min), best(x[1], all, min), best(xx[1], all, min))
        table1.add_hline(start=2)
        all = [1-x[2]-x[3], 1-x[0]-x[1], 1-xx[0]-xx[1]]
        table1.add_row('', italic('unassigned'), f.format(all[0]), f.format(all[1]), f.format(all[2]))
        table1.add_hline()

    doc.append(table1)

    table1.generate_tex(OUT_WD+'/tables/'+name)
    # doc.generate_pdf(clean_tex=False, filepath=OUT_WD+'/'+)

def results2latex2(name=''):
    wd = RESULTS_WD + '/id_assignment/'

    from pylatex import Document, Section, Subsection, Tabular, Tabularx, MultiColumn, MultiRow
    from pylatex.utils import bold, italic, verbatim, escape_latex, NoEscape
    from pylatex.package import Package
    with open(DEV_WD+ '/thesis/results/overall.pkl') as f:
        results = pickle.load(f)

    keys = ['Cam1', 'Zebrafish', 'Camera3', 'Sowbug3']
    # for key in keys:
    for pname in project_real_names.keys():
        try:
            with open(wd+name+'_'+pname) as f:
                r = pickle.load(f)

            cc = r[1][0]['cc']
            mc = r[1][0]['mc']
        except:
            pass

        results[pname] = (results[pname][0], results[pname][1], cc, mc)
        results[pname+'_nogaps'] = (results[pname+'_nogaps'][0], results[pname+'_nogaps'][1], cc, mc)

    doc = Document("multirow")
    doc.packages.add(Package('xcolor', options='table, dvipsnames'))
    # doc.append(Package('xcolors', options='table'))


    # table1 = Tabular('|c|c|c|c|', booktabs=True)
    table1 = Tabular('|c|c||c|c|c|')
    table1.add_hline()
    table1.add_row('', '', bold('FERDA'), bold('idTracker'), bold('idTracker I'))
    table1.add_hline()

    f = FORMAT_PERCENTS
    for key in keys:
        x = results[key]
        xx = results[key+'_nogaps']
        table1.add_hline()

        all = [x[2], x[0], xx[0]]
        table1.add_row((MultiRow(3, data=bold(key)), italic('correct'), best(x[2], all), best(x[0], all), best(xx[0], all)))
        table1.add_hline(start=2)
        all = [x[3], x[1], xx[1]]
        table1.add_row('', italic('wrong'), best(x[3], all, min), best(x[1], all, min), best(xx[1], all, min))
        table1.add_hline(start=2)
        all = [1-x[2]-x[3], 1-x[0]-x[1], 1-xx[0]-xx[1]]
        table1.add_row('', italic('unassigned'), f.format(all[0]), f.format(all[1]), f.format(all[2]))
        table1.add_hline()

    doc.append(table1)

    table1.generate_tex(OUT_WD+'/tables/overall_'+name)


if __name__ == '__main__':
    wd = RESULTS_WD+'/id_assignment/'


    new_ = ['lp_clean',
                'lp_SEG',
                'lp_IDCR_f',
                'lp_IDCR_full',
                'lp_SEG_IDCR_full']

    old_ = ['lp_id',
                'lp_id_SEG',
                'lp_id_SEG_IDCR',
                'lp_id_IDCR_HIL',
                'lp_id_IDCR',
                'lp_HIL_INIT',
                'lp_HIL_INIT2',
                'lp_HIL_INIT3',
                'lp_HIL_INIT_10',
                'lp_HIL_INIT_SEG',
                'lp_HIL_INIT_IDCR',
                'lp_HIL_INIT_SEG_IDCR',
                'lp_HIL_INIT_SEG_IDCR2']

    for test in old_:
        print("$$$$$$$$$$$", test, "$$$$$$$$")
        for pname in project_real_names.keys():
            try:
                with open(wd+test+'_'+pname) as f:
                    r = pickle.load(f)

                cc = r[1][0]['cc']
                mc = r[1][0]['mc']
                if pname[:4] == 'Cam1':
                    ccdif = cc-0.7168
                    mcdif = mc-0.0032
                elif pname[:4] == 'Zebr':
                    ccdif = cc-0.8800
                    mcdif = mc-0.0063
                elif pname[:4] == 'Sowb':
                    ccdif = cc-0.7060
                    mcdif = mc-0.1528
                else:
                    ccdif = cc-0.8238
                    mcdif = mc-0.0523

                while len(pname) < 10:
                    pname = pname+' '

                udec = None
                try:
                    udec = r[1][0]['HIL_INIT']
                except:
                    pass

                print(" {}\t\t{:.2%}({:+.2%})\t {:.2%}({:+.2%}) \t\t #HiLs: {}, #UD: {}".format(pname, cc, ccdif, mc, mcdif, r[1][0]['HIL'], udec))

                # print r[0]
            except:
                print("\tNOT READY YET")

    # run(semistate='lp_id_0', dir_name='overall_clean')
    # results2latex2('lp_id')
    # run(semistate='lp_id_IDCR_0', dir_name='overall_IDCR')
    # results2latex2('lp_id_IDCR')


    # run(semistate='lp_HIL_INIT3_0', dir_name='lp_HIL_INIT3')
    results2latex2('lp_HIL_INIT3')

    # results2latex2('lp_HIL_INIT')


    # run(semistate='lp_HIL_INIT_0', dir_name='overall_HIL_INIT')



    #
    # for fn in ['Cam1_playground2017-01-04_19-04-31',
    #            'Sowbug32017-01-02_15-20-56',
    #            'zebrafish_playground2017-01-02_20-37-12',
    #            'Camera32017-01-04_19-39-45']:
    #
    #     print fn
    #     with open(wd+fn) as f:
    #         r = pickle.load(f)
    #
    #         for it in r[1]:
    #             if isinstance(it, tuple):
    #                 print "{:.2%} {:.2%}".format(it[0], it[1])
    #             else:
    #                 print "{:.2%} {:.2%}".format(it['cc'], it['mc'])


    # run(semistate='id_classified_HIL_init_0', dir_name='overall_HIL_init')
    # run(semistate='id_classified_no_HIL', dir_name='overall_no_HIL', HIL=False)
    # results2latex('lp_id_0.pkl')