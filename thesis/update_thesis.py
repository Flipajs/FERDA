from __future__ import unicode_literals
import os, shutil

RESULTS_WD = '/Users/flipajs/Documents/dev/ferda/thesis'
THESIS_WD = '/Users/flipajs/Dropbox/SCHOOL/5th_semester/thesis/master-thesis'

#images
shutil.rmtree(THESIS_WD+'/imgs/auto')
shutil.copytree(RESULTS_WD+'/out/imgs', THESIS_WD+'/imgs/auto')

#tables
shutil.rmtree(THESIS_WD+'/tables/auto')
shutil.copytree(RESULTS_WD+'/out/tables', THESIS_WD+'/tables/auto')