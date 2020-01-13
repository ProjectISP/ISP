import os
import shutil
from distutils.core import setup
from setuptools.command.build_ext import build_ext

import numpy
from Cython.Build import cythonize
import subprocess as sb

from isp import ROOT_DIR

cy_path = os.path.join(ROOT_DIR, 'cython_code')


def extract_nll():
    extract_at = os.path.join(ROOT_DIR, "NLL7")
    try:
        if not os.path.isdir(extract_at):
            os.mkdir(extract_at)
        else:
            shutil.rmtree(extract_at)
            os.mkdir(extract_at)
        shutil.unpack_archive("NLL7.00_src.tgz", extract_at)
        os.mkdir(os.path.join(extract_at, "bin"))
    except IOError as error:
        print(error)
    return extract_at


class CustomBuildExtCommand(build_ext):
    def run(self):
        nll_dir = extract_nll()
        self.make_nll(nll_dir)
        build_ext.run(self)

    def make_nll(self, nll_dir):
        src_path = os.path.join(nll_dir, "src")
        bin_path = os.path.join(nll_dir, "bin")
        command = "export MYBIN={bin_path}; cd {src_path}; make -R all ".format(bin_path=bin_path, src_path=src_path)
        with sb.Popen(command, stdout=sb.PIPE, shell=True) as process:
            proc_stdout = process.communicate()[0].strip()
            print(proc_stdout)
            print("Makefile for nll done.")


setup(
    cmdclass={
        'build_ext': CustomBuildExtCommand,
    },
    ext_modules=cythonize(os.path.join(cy_path, 'ccwt_cy.pyx')),
    include_dirs=[numpy.get_include()],
    ext_package='isp.c_lib'
)
