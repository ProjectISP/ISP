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

    try:
        folder_nll = os.path.join(ROOT_DIR,"NLL7")
        if os.path.isdir(folder_nll):
            shutil.rmtree(folder_nll)
    except:
        pass

    extract_at = os.path.join(ROOT_DIR, "NLL7")
    try:
        if not os.path.isdir(extract_at):
            os.mkdir(extract_at)
        else:
            shutil.rmtree(extract_at)
            os.mkdir(extract_at)
        shutil.unpack_archive("NLL2023_src.tgz", extract_at)
        os.mkdir(os.path.join(extract_at, "bin"))
    except IOError as error:
        print("Warnning", error)
    return extract_at

def extract_focmec():

    try:
        folder_focmec = os.path.join(ROOT_DIR,"focmec")
        if os.path.isdir(folder_focmec):
            shutil.rmtree(folder_focmec)
    except:
        pass

    check_focmec = os.path.join(ROOT_DIR, "focmec")
    try:
        if not os.path.isdir(check_focmec):
            shutil.unpack_archive("focmec2023_src.tgz", ROOT_DIR)
        else:
            shutil.rmtree(check_focmec)
            shutil.unpack_archive("focmec2023_src.tgz", ROOT_DIR)
        
    except IOError as error:
        print(error)
    return check_focmec

def extract_mti():

    try:
        folder_mti = os.path.join(ROOT_DIR, "mti","green_source")
        if os.path.isdir(folder_mti):
            shutil.rmtree(folder_mti)
    except:
        pass

    extract_at = os.path.join(ROOT_DIR, "mti")
    print("mti_folder", extract_at)

    try:
        if not os.path.isdir(extract_at):
            os.mkdir(extract_at)
            shutil.unpack_archive("isola2023_src.tgz", extract_at)
        else:
            shutil.unpack_archive("isola2023_src.tgz", extract_at)
            #shutil.rmtree(extract_at)
            #os.mkdir(extract_at)
        
    except IOError as error:
        print(error)
    return extract_at


class CustomBuildExtCommand(build_ext):
    def run(self):


        print("Extracting files")
        nll_dir = extract_nll()
        focmec_dir = extract_focmec()
        mti_dir = extract_mti()
        print("Finished extraction")

        print("Start compiling")
        self.make_nll(nll_dir)
        self.make_focmec(focmec_dir)
        self.make_mti(mti_dir)
        print("Finished Compilation")
        # run build
        build_ext.run(self)

    def make_nll(self, nll_dir):
        src_path = os.path.join(nll_dir, "src")
        bin_path = os.path.join(nll_dir, "bin")
        command = "export MYBIN={bin_path}; cd {src_path}; sh compile_nll.sh".format(bin_path=bin_path, src_path=src_path)
        try:
            exc_cmd(command, shell=True)
            print("NLL7 successfully installed")
        except sb.CalledProcessError as e:  # this is a bad error/warning.
            print("Warnning on trying to run nll make file / NLL7 successfully installed")
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

        green_path = os.path.join(mti_dir,"green_source")
        command = "sh compile_mti_ifort.sh"

        try:
            exc_cmd(command, cwd=green_path, shell=True)
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
    ext_modules=cythonize(os.path.join(cy_path, 'ccwt_cy.pyx'), os.path.join(cy_path, 'whiten.pyx')),
    include_dirs=[numpy.get_include()],
    ext_package='isp.c_lib'
)
