import os
import shutil
from distutils.core import setup
from setuptools.command.build_ext import build_ext

import numpy
from Cython.Build import cythonize
import subprocess as sb

from isp import ROOT_DIR

cy_path = os.path.join(ROOT_DIR, 'cython_code')

def exc_cmd(cmd, **kwargs):
    """
    Execute a subprocess Popen and catch except.

    :param cmd: The command to be execute.
    :param kwargs: All subprocess.Popen(..).
    :return: The stdout.
    :except excepts: SubprocessError, CalledProcessError

    :keyword stdin: Default=subprocess.PIPE
    :keyword stdout: Default=subprocess.PIPE
    :keyword stderr: Default=subprocess.PIPE
    :keyword encoding: Default=utf-8
    """
    stdout = kwargs.pop("stdout", sb.PIPE)
    stdin = kwargs.pop("stdin", sb.PIPE)
    stderr = kwargs.pop("stderr", sb.PIPE)
    encoding = kwargs.pop("encoding", "utf-8")

    with sb.Popen(cmd, stdin=stdin, stdout=stdout, stderr=stderr, encoding=encoding, **kwargs) as p:
        try:
            std_out, std_err = p.communicate(timeout=15)
        except sb.TimeoutExpired:
            p.kill()
            std_out, std_err = p.communicate()  # try again if timeout fails.
        if p.returncode != 0:  # Bad error.
            raise sb.CalledProcessError(p.returncode, std_err.strip())
        elif len(std_err) != 0:  # Some possible errors trowed by the running subprocess, but not critical.
            raise sb.SubprocessError(std_err.strip())
        return std_out.strip()

class CustomBuildExtCommand(build_ext):
    def run(self):
        # run build
        build_ext.run(self)



setup(
    cmdclass={
        'build_ext': CustomBuildExtCommand,
    },
    name="whiten app",
    ext_modules=cythonize(os.path.join(cy_path, 'whiten.pyx')),
    include_dirs=[numpy.get_include()],
    ext_package='isp.c_lib',zip_safe=False
)
