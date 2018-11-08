from __future__ import division
from __future__ import unicode_literals
from builtins import str
from builtins import range
from past.utils import old_div
import os
import sys

LIMIT = 1000

def generate_html(fnames, d, out_name):
    fs = ""
    fs += '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN"' + '\n'
    fs += '    "http://www.w3.org/TR/html4/loose.dtd">' + '\n'
    fs += '<html>' + '\n'
    fs += '<title>results</title>' + '\n'
    fs += '<head>' + '\n'
    fs += '</head>' + '\n'
    fs += '<body>' + '\n'
    fs += '<h1>' + out_name + '</h1>'
    try:
        for fname in fnames:
            fs += '<img src="' + d + '/' + fname + '" />'
    except OSError:
        pass

    fs += '</body>' + '\n'
    fs += '</html>'

    with open(DIR + '/' + out_name + '.html', 'wb') as f:
        f.write(fs)

def make_web(DIR, data=''):
    dirs = sorted(os.listdir(DIR))

    for d in dirs:
        if not os.path.isdir(DIR+'/'+d):
            continue

        fnames = sorted(os.listdir(DIR + '/' + d))
        for i in range(old_div(len(fnames),LIMIT)):
            str_i = str(i)
            if len(str_i) == 1:
                str_i = "0"+str_i

            generate_html(fnames[LIMIT*i:min(LIMIT*(i+1), len(fnames))], d, d+'_part'+str_i)

if __name__ == '__main__':
    # DIR = '/Volumes/Seagate Expansion Drive/CNN_HH1_train'
    DIR = '/Users/flipajs/Documents/dev/ferda/scripts/gaster_grooming_out'
    make_web(DIR)

