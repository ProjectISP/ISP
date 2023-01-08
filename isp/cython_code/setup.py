from setuptools import setup
from Cython.Build import cythonize

setup(
    name='whiten app',
    ext_modules=cythonize("whiten.pyx"),
    zip_safe=False,
)