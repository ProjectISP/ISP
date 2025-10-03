import os
import shutil
import warnings
import platform
import subprocess as sb
from os.path import isfile, join
from os import listdir
import numpy
from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
from Cython.Compiler.Errors import CompileError


try:
    from isp import ROOT_DIR
except Exception as e:
    raise RuntimeError(f"Could not import isp. Ensure ISP is importable. Error: {e}")

system_computer = platform.system().lower()
machine_computer = platform.machine().lower()

if system_computer in ("linux", "linux2"):
    system_path = os.path.join(ROOT_DIR, "linux_bin")
elif system_computer == "darwin":
    if machine_computer == "arm64":
        system_path = os.path.join(ROOT_DIR, "mac_m_bin")  # Apple Silicon
    else:
        system_path = os.path.join(ROOT_DIR, "mac_bin")    # Intel mac
else:
    # Unsupported platforms won't stop the build; just print info.
    print(f"Unsupported system: {system_computer} ({machine_computer})")
    system_path = os.path.join(ROOT_DIR, "bin_unknown")

cy_path = os.path.join(ROOT_DIR, "cython_code")

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

    try:
        folder_work_mti = os.path.join(ROOT_DIR, "mti","green")
        if os.path.isdir(older_work_mti):
            shutil.rmtree(older_work_mti)
    except:
        pass

    extract_at = os.path.join(ROOT_DIR, "mti")

    try:
        if not os.path.isdir(extract_at):
            os.mkdir(extract_at)
            shutil.unpack_archive("isola2023_src.tgz", extract_at)
        else:
            shutil.unpack_archive("isola2023_src.tgz", extract_at)

        if not os.path.isdir(folder_work_mti):
            os.mkdir(folder_work_mti)

        
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
        work_dir = os.path.join(mti_dir,"green")
        command = "sh compile_mti_ifort.sh"

        try:
            exc_cmd(command, cwd=green_path, shell=True)
            print("MTI successfully installed")
            self.copy_mti_binaries(green_path, work_dir)

        except sb.CalledProcessError as e:  # this is a bad error.
            print("Error on trying to run MTI make file.")
            print(e)
        except sb.SubprocessError:  # some warnings nothing so bad.
            print("MTI successfully installed")


    def copy_mti_binaries(self, source_code, destination):

        allfiles = [f for f in listdir(source_code) if isfile(join(source_code, f))]
        # iterate on all files to move them to destination folder
        for f in allfiles:
            src_path = os.path.join(source_code, f)
            dst_path = os.path.join(destination, f)
            shutil.copy(src_path, dst_path)


# ----------------------------------------------------------------------
# Build Cython extensions with per-module try/except
# ----------------------------------------------------------------------
extra_compile_args = ["/O2"] if system_computer == "windows" else ["-O3"]
extra_link_args = []
extra_libraries = [] if system_computer == "windows" else ["m"]  # libm on POSIX

def try_add_extension(modname, src):
    """
    Try to cythonize a single extension. On failure, warn and continue.
    Returns a list (possibly empty) of cythonized Extension objects.
    """
    try:
        ext = Extension(
            name=modname,
            sources=[src],
            include_dirs=[numpy.get_include(), cy_path],
            language="c",
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            libraries=extra_libraries,
        )
        return cythonize(
            [ext],
            language_level=3,
            compiler_directives={
                "boundscheck": False,
                "wraparound": False,
                "cdivision": True,
                "nonecheck": False,
            },
        )
    except CompileError as e:
        warnings.warn(f"Could not compile {modname}: {e}. Skipping.")
    except Exception as e:
        warnings.warn(f"Unexpected error compiling {modname}: {e}. Skipping.")
    return []

# Collect target .pyx modules (explicit list; add/remove as needed)
# cy_modules = [
#     ("isp.cython_code.hampel",         os.path.join(cy_path, "hampel.pyx")),
#     # CF modules:
#     ("isp.cython_code.rec_filter",     os.path.join(cy_path, "rec_filter.pyx")),
#     ("isp.cython_code.lib_rec_hos",    os.path.join(cy_path, "lib_rec_hos.pyx")),
#     ("isp.cython_code.lib_rec_rms",    os.path.join(cy_path, "lib_rec_rms.pyx")),
#     ("isp.cython_code.lib_rosenberger",os.path.join(cy_path, "lib_rosenberger.pyx")),
#     ("isp.cython_code.lib_rec_cc",     os.path.join(cy_path, "lib_rec_cc.pyx")),
#     # Optional/problematic ones (won't break the build if they fail):
#     ("isp.cython_code.ccwt_cy",        os.path.join(cy_path, "ccwt_cy.pyx")),
#     ("isp.cython_code.whiten",         os.path.join(cy_path, "whiten.pyx")),
# ]
cy_modules = [
     ("isp.cython_code.hampel",         os.path.join(cy_path, "hampel.pyx")),
     ("isp.cython_code.ccwt_cy",        os.path.join(cy_path, "ccwt_cy.pyx")),
     ("isp.cython_code.whiten",         os.path.join(cy_path, "whiten.pyx"))]

ext_list = []
for modname, src in cy_modules:
    if os.path.isfile(src):
        ext_list.extend(try_add_extension(modname, src))
    else:
        warnings.warn(f"Source not found for {modname}: {src}. Skipping.")

# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------
setup(
    cmdclass={'build_ext': CustomBuildExtCommand},
    name='isp_package',
    version='2.0',
    description='ISP setup script',
    packages=find_packages(include=['isp', 'isp.*']),
    include_dirs=[numpy.get_include()],
    ext_modules=ext_list,
)

#setup(
#    cmdclass={
#        'build_ext': CustomBuildExtCommand,
#    },
#    ext_modules=cythonize(os.path.join(cy_path, 'ccwt_cy.pyx')),
#    include_dirs=[numpy.get_include()],
#    ext_package='isp.c_lib'
#)
