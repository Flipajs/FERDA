from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
import sys
import subprocess
import os
import datetime
import pipes
import pickle as pickle


# from http://stackoverflow.com/a/14392472/3074835  by Dietrich Epp
def exists_remote(host, path):
    proc = subprocess.Popen(
        ['ssh', host, 'test -e %s' % pipes.quote(path)])
    proc.wait()
    return proc.returncode == 0


def exists_local_directory(path):
    if not os.path.exists(path):
        os.mkdir(path)


def longDate():
    Y = format(datetime.datetime.now().year, '04')
    M = format(datetime.datetime.now().month, '02')
    D = format(datetime.datetime.now().day, '02')
    h = format(datetime.datetime.now().hour, '02')
    m = format(datetime.datetime.now().minute, '02')
    s = format(datetime.datetime.now().second, '02')

    return Y + M + D + "_" + h + m + s


def fixPath(path):
    if (path[len(path) - 1] == "/"):
        ret = path;
    else:
        ret = path + "/";

    return ret;


def get_videopath(project_FileName):
    d = pickle.load(open(project_FileName, 'rb'))
    return d['video_paths']
