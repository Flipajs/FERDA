import os, shutil

RESULTS_WD = '/Users/flipajs/Documents/dev/ferda/thesis'
THESIS_WD = '/Users/flipajs/Dropbox/SCHOOL/5th_semester/thesis/master-thesis'
shutil.rmtree(THESIS_WD+'/imgs/auto')
shutil.copytree(RESULTS_WD+'/out/imgs', THESIS_WD+'/imgs/auto')

