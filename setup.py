import os
from distutils.core import setup

import numpy
from Cython.Build import cythonize

from isp import ROOT_DIR

cy_path = os.path.join(ROOT_DIR, 'cython_code')

setup(
    ext_modules=cythonize(os.path.join(cy_path, 'ccwt_cy.pyx')),
    include_dirs=[numpy.get_include()],
    ext_package='isp.c_lib'
)
