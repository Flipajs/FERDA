__author__ = 'flipajs'

import platform

def __bootstrap__():
    global __bootstrap__, __loader__, __file__
    import pkg_resources, imp

    if platform.system() == 'Darwin':
        __file__ = pkg_resources.resource_filename(__name__, 'maxflow_wrapper/cyMaxflow-OSX.so')

    if platform.system() == 'Linux':
        __file__ = pkg_resources.resource_filename(__name__, 'maxflow_wrapper/cyMaxflow_KUBUNTU64.so')
        
    __loader__ = None
    
    del __bootstrap__, __loader__
    imp.load_dynamic(__name__, __file__)

__bootstrap__()
