__author__ = 'filip@naiser.cz'

from utils.misc import is_flipajs_pc
import platform


def __bootstrap__():
    global __bootstrap__, __loader__, __file__
    import sys, pkg_resources, imp

    if platform.system() == 'Darwin':
        __file__ = pkg_resources.resource_filename(__name__, 'cyMser_64_OSX.so')
    elif platform.system() == 'Linux':
        __file__ = pkg_resources.resource_filename(__name__, 'cyMser_ubuntu.so')

    __loader__ = None;
    del __bootstrap__, __loader__
    imp.load_dynamic(__name__, __file__)


__bootstrap__()