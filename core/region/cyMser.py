__author__ = 'filip@naiser.cz'

import platform


def __bootstrap__():
    global __bootstrap__, __loader__, __file__
    import sys, pkg_resources, imp

    if platform.system() == 'Darwin':
        __file__ = pkg_resources.resource_filename(__name__, 'libs/cyMser_64_OSX.so')
    elif platform.system() == 'Linux':
        if platform.architecture()[0] == '64bit':
            __file__ = pkg_resources.resource_filename(__name__, 'libs/cyMser_debian.so')
        else:
            __file__ = pkg_resources.resource_filename(__name__, 'libs/cyMser_ubuntu32.so')

    __loader__ = None;
    del __bootstrap__, __loader__
    imp.load_dynamic(__name__, __file__)

__bootstrap__()