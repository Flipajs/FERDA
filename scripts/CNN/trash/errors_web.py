from __future__ import unicode_literals
import os
import sys

def make_web(DIR, data=''):
    title = "errors"
    fs = ""
    fs += '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN"' + '\n'
    fs += '    "http://www.w3.org/TR/html4/loose.dtd">' + '\n'
    fs += '<html>' + '\n'
    fs += '<title>' + title + '</title>' + '\n'
    fs += '<head>' + '\n'
    fs += '</head>' + '\n'
    fs += '<body>' + '\n'

    fs += '<h1>'
    fs += "{}".format(data)
    fs += '</h1>'

    dirs = sorted(os.listdir(DIR))

    for d in dirs:
        fs += '<h1>' + d + '</h1>'
        try:
            for fname in os.listdir(DIR + '/' + d):
                fs += '<img src="' + d + '/' + fname + '" />'
        except OSError:
            pass

    fs += '</body>' + '\n'
    fs += '</html>'

    with open(DIR + '/index.html', 'wb') as f:
        f.write(fs)

if __name__ == '__main__':
    DIR = sys.argv[1]

    make_web(DIR)

