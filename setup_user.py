import os
import shutil
from distutils.core import setup
from setuptools.command.build_ext import build_ext
from os.path import isfile, join
from os import listdir
import numpy
from Cython.Build import cythonize
import subprocess as sb
from setuptools import setup, find_packages
from isp import ROOT_DIR
from sys import platform

if platform == "linux" or platform == "linux2":
    system_path = os.path.join(ROOT_DIR,"linux_bin")
    print(system_path)
elif platform == "darwin":
    system_path = os.path.join(ROOT_DIR,"mac_bin")
    print(system_path)


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


def nll_create_folders():
    
    try:
        folder_nll = os.path.join(ROOT_DIR,"NLL7")
        if os.path.isdir(folder_nll):
            shutil.rmtree(folder_nll)
    except:
        pass

    folder_create = os.path.join(ROOT_DIR,"NLL7","src","bin")
    source_code = os.path.join(system_path, "NLL_new")
    print(system_path, source_code, folder_create)
    try:
        #shutil.rmtree(folder_create)
        os.makedirs(folder_create)
    except IOError as error:
        print(error)
    return source_code, folder_create

def focmec_create_folders():


    try:
        folder_focmec = os.path.join(ROOT_DIR, "focmec")
        if os.path.isdir(folder_focmec):
            shutil.rmtree(folder_focmec)
    except:
        pass


    folder_create = os.path.join(ROOT_DIR,"focmec","bin")
    source_code = os.path.join(system_path, "FOCMEC/bin")
    print(system_path, source_code, folder_create)
    try:
        os.makedirs(folder_create)
    except IOError as error:
        print(error)
    return source_code, folder_create


def mti_create_folders():

    try:
        folder_mti = os.path.join(ROOT_DIR, "mti","green_source")
        if os.path.isdir(folder_mti):
            shutil.rmtree(folder_mti)
    except:
        pass


    try:
        folder_work_mti = os.path.join(ROOT_DIR, "mti","green")
        if os.path.isdir(folder_work_mti):
            shutil.rmtree(folder_work_mti)
    except:
        pass

    folder_work_create = os.path.join(ROOT_DIR,"mti","green")
    folder_create = os.path.join(ROOT_DIR,"mti","green_source")
    source_code = os.path.join(system_path, "mti_green")
    print(system_path, source_code, folder_create)
    try:
        #shutil.rmtree(folder_create)
        os.makedirs(folder_create)
        os.makedirs(folder_work_create)
    except IOError as error:
        print(error)
    return source_code, folder_create, folder_work_create

class CustomBuildExtCommand(build_ext):
    def run(self):

        
        print("Creating 3th package folders")
        source_code, destination = nll_create_folders()
        self.copy_nll_binaries(source_code, destination)

        source_code, destination = focmec_create_folders()
        self.copy_focmec_binaries(source_code, destination)

        source_code, destination, work_dir= mti_create_folders()
        self.copy_mti_binaries(source_code, destination)
        self.copy_mti_binaries(source_code, work_dir)

        # run cython build
        build_ext.run(self)

    def copy_nll_binaries(self, source_code, destination):

        print("gather all NLL files")
        allfiles = [f for f in listdir(source_code) if isfile(join(source_code, f))]
        print(allfiles)
        # iterate on all files to move them to destination folder
        for f in allfiles:
            src_path = os.path.join(source_code, f)
            dst_path = os.path.join(destination, f)
            print(src_path, dst_path)
            shutil.copy(src_path, dst_path)

    def copy_focmec_binaries(self, source_code, destination):

        print("gather all focmec files")
        allfiles = [f for f in listdir(source_code) if isfile(join(source_code, f))]
        print(allfiles)
        # iterate on all files to move them to destination folder
        for f in allfiles:
            src_path = os.path.join(source_code, f)
            dst_path = os.path.join(destination, f)
            print(src_path, dst_path)
            shutil.copy(src_path, dst_path)

    def copy_mti_binaries(self, source_code, destination):

        print("gather all isola files")
        allfiles = [f for f in listdir(source_code) if isfile(join(source_code, f))]
        print(allfiles)
        # iterate on all files to move them to destination folder
        for f in allfiles:
            src_path = os.path.join(source_code, f)
            dst_path = os.path.join(destination, f)
            print(src_path, dst_path)
            shutil.copy(src_path, dst_path)

    
setup(
    cmdclass={'build_ext': CustomBuildExtCommand,},
    name='isp_package',
    version='2.0',
    description='ISP setup script',
    packages=find_packages(include=['isp', 'isp.*']),
    include_dirs=[numpy.get_include()])
