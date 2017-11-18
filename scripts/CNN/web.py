import os
import sys

def make_web(DIR, data=''):
    dirs = sorted(os.listdir(DIR))

    for d in dirs:
        fs = ""
        fs += '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN"' + '\n'
        fs += '    "http://www.w3.org/TR/html4/loose.dtd">' + '\n'
        fs += '<html>' + '\n'
        fs += '<title>results</title>' + '\n'
        fs += '<head>' + '\n'
        fs += '</head>' + '\n'
        fs += '<body>' + '\n'
        fs += '<h1>' + d + '</h1>'
        try:
            fnames = sorted(os.listdir(DIR+'/'+d))
            for fname in fnames:
                fs += '<img src="' + d + '/' + fname + '" />'
        except OSError:
            pass

        fs += '</body>' + '\n'
        fs += '</html>'

        with open(DIR + '/'+d+'.html', 'wb') as f:
            f.write(fs)

if __name__ == '__main__':
    # DIR = '/Volumes/Seagate Expansion Drive/CNN_HH1_train'
    DIR = '/Users/flipajs/Documents/dev/ferda/scripts/out9'
    make_web(DIR)

