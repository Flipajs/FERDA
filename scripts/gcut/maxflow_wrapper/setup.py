from __future__ import unicode_literals
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import sys

extensions = [
    Extension("cyMaxflow", ["cyMaxflow.pyx", "maxflow.cpp", "../maxflow-v3.03.src/graph.cpp", "../maxflow-v3.03.src/maxflow.cpp"],
        include_dirs = ["../maxflow_wrapper", "../maxflow-v3.03.src"],
        # libraries = ["maxflow_wrapper"],
        # library_dirs = ["../maxflow_wrapper", "../maxflow-v3.03.src"]
    	)
]


setup(
    name = "My hello app",
    ext_modules = cythonize(extensions),
)