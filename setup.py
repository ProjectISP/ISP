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


def extract_focmec():
    extract_at = os.path.join(ROOT_DIR, "FOCMEC")
    try:
        if not os.path.isdir(extract_at):
            os.mkdir(extract_at)
        else:
            shutil.rmtree(extract_at)
            os.mkdir(extract_at)
        shutil.unpack_archive("focmec.tar", extract_at)
    except IOError as error:
        print(error)
    return extract_at

def extract_mti():
    extract_at = os.path.join(ROOT_DIR, "mti")
    try:
        if not os.path.isdir(extract_at):
            os.mkdir(extract_at)
        else:
            shutil.rmtree(extract_at)
            os.mkdir(extract_at)
        shutil.unpack_archive("mti.tar.gz", extract_at)
    except IOError as error:
        print(error)
    return extract_at


class CustomBuildExtCommand(build_ext):
    def run(self):
        print("Extracting files.")
        #nll_dir = extract_nll()
        #focmec_dir = extract_focmec()
        mti_dir = extract_mti()
        print("Finished ")

        #self.make_nll(nll_dir)
        #self.make_focmec(focmec_dir)
        self.make_mti(mti_dir)

        # run build
        build_ext.run(self)

    def make_nll(self, nll_dir):
        src_path = os.path.join(nll_dir, "src")
        bin_path = os.path.join(nll_dir, "bin")
        command = "export MYBIN={bin_path}; cd {src_path}; make -R all ".format(bin_path=bin_path, src_path=src_path)
        try:
            exc_cmd(command, shell=True)
            print("NLL7 successfully installed")
        except sb.CalledProcessError as e:  # this is a bad error.
            print("Error on trying to run nll make file.")
            print(e)
        except sb.SubprocessError:  # some warnings nothing so bad.
            print("NLL7 successfully installed")

    def make_focmec(self, focmec_dir):
        src_path = os.path.join(focmec_dir, "src")
        command = "sh build_package_sh"
        try:
            exc_cmd(command, cwd=src_path, shell=True)
            print("FOCMEC successfully installed")
        except sb.CalledProcessError as e:  # this is a bad error.
            print("Error on trying to run focmec make file.")
            print(e)
        except sb.SubprocessError:  # some warnings nothing so bad.
            print("FOCMEC successfully installed")

    def make_mti(self, mti_dir):
        src_path = os.path.join(mti_dir, "green")
        command = "sh compile_mti.sh"
        try:
            exc_cmd(command, cwd=src_path, shell=True)
            print("MTI successfully installed")
        except sb.CalledProcessError as e:  # this is a bad error.
            print("Error on trying to run MTI make file.")
            print(e)
        except sb.SubprocessError:  # some warnings nothing so bad.
            print("MTI successfully installed")


setup(
    cmdclass={
        'build_ext': CustomBuildExtCommand,
    },
    #ext_modules=cythonize(os.path.join(cy_path, 'ccwt_cy.pyx')),
    #include_dirs=[numpy.get_include()],
    #ext_package='isp.c_lib'
)
