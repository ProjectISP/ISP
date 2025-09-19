#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

# ----------------------------------------------------------------------
# Paths / environment
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# Helper to run subprocess safely
# ----------------------------------------------------------------------
def exc_cmd(cmd, **kwargs):
    stdout = kwargs.pop("stdout", sb.PIPE)
    stdin = kwargs.pop("stdin", sb.PIPE)
    stderr = kwargs.pop("stderr", sb.PIPE)
    encoding = kwargs.pop("encoding", "utf-8")

    with sb.Popen(cmd, stdin=stdin, stdout=stdout, stderr=stderr, encoding=encoding, **kwargs) as p:
        try:
            std_out, std_err = p.communicate(timeout=15)
        except sb.TimeoutExpired:
            p.kill()
            std_out, std_err = p.communicate()
        if p.returncode != 0:
            raise sb.CalledProcessError(p.returncode, std_err.strip())
        elif len(std_err) != 0:
            raise sb.SubprocessError(std_err.strip())
        return std_out.strip()

# ----------------------------------------------------------------------
# Folder preparation for third-party binaries (NLL7, FOCMEC, MTI)
# ----------------------------------------------------------------------
def nll_create_folders():
    try:
        folder_nll = os.path.join(ROOT_DIR, "NLL7")
        if os.path.isdir(folder_nll):
            shutil.rmtree(folder_nll)
    except Exception:
        pass

    folder_create = os.path.join(ROOT_DIR, "NLL7", "src", "bin")
    source_code = os.path.join(system_path, "NLL_new")
    try:
        os.makedirs(folder_create, exist_ok=True)
    except IOError as error:
        print(error)
    return source_code, folder_create

def focmec_create_folders():
    try:
        folder_focmec = os.path.join(ROOT_DIR, "focmec")
        if os.path.isdir(folder_focmec):
            shutil.rmtree(folder_focmec)
    except Exception:
        pass

    folder_create = os.path.join(ROOT_DIR, "focmec", "bin")
    source_code = os.path.join(system_path, "FOCMEC", "bin")
    try:
        os.makedirs(folder_create, exist_ok=True)
    except IOError as error:
        print(error)
    return source_code, folder_create

def mti_create_folders():
    try:
        folder_mti = os.path.join(ROOT_DIR, "mti", "green_source")
        if os.path.isdir(folder_mti):
            shutil.rmtree(folder_mti)
    except Exception:
        pass

    try:
        folder_work_mti = os.path.join(ROOT_DIR, "mti", "green")
        if os.path.isdir(folder_work_mti):
            shutil.rmtree(folder_work_mti)
    except Exception:
        pass

    folder_work_create = os.path.join(ROOT_DIR, "mti", "green")
    folder_create = os.path.join(ROOT_DIR, "mti", "green_source")
    source_code = os.path.join(system_path, "mti_green")
    try:
        os.makedirs(folder_create, exist_ok=True)
        os.makedirs(folder_work_create, exist_ok=True)
    except IOError as error:
        print(error)
    return source_code, folder_create, folder_work_create

# ----------------------------------------------------------------------
# Custom build command to copy binaries + build Cython
# ----------------------------------------------------------------------
class CustomBuildExtCommand(build_ext):
    def run(self):
        print("Creating 3rd-party package folders & copying binaries...")

        src, dst = nll_create_folders()
        self._copy_binaries(src, dst, label="NLL")

        src, dst = focmec_create_folders()
        self._copy_binaries(src, dst, label="FOCMEC")

        src, dst, work = mti_create_folders()
        self._copy_binaries(src, dst, label="MTI")
        self._copy_binaries(src, work, label="MTI(work)")

        # Now run normal Cython build
        super().run()

    def _copy_binaries(self, source_code, destination, label=""):
        print(f"Gathering {label} files from: {source_code}")
        try:
            allfiles = [f for f in listdir(source_code) if isfile(join(source_code, f))]
        except Exception as e:
            warnings.warn(f"Could not list {label} binaries at {source_code}: {e}")
            return
        for f in allfiles:
            src_path = os.path.join(source_code, f)
            dst_path = os.path.join(destination, f)
            try:
                shutil.copy(src_path, dst_path)
                print(f"  + {src_path} -> {dst_path}")
            except Exception as e:
                warnings.warn(f"Failed to copy {src_path} to {dst_path}: {e}")

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
cy_modules = [
    ("isp.cython_code.hampel",         os.path.join(cy_path, "hampel.pyx")),
    # CF modules:
    ("isp.cython_code.rec_filter",     os.path.join(cy_path, "rec_filter.pyx")),
    ("isp.cython_code.lib_rec_hos",    os.path.join(cy_path, "lib_rec_hos.pyx")),
    ("isp.cython_code.lib_rec_rms",    os.path.join(cy_path, "lib_rec_rms.pyx")),
    ("isp.cython_code.lib_rosenberger",os.path.join(cy_path, "lib_rosenberger.pyx")),
    ("isp.cython_code.lib_rec_cc",     os.path.join(cy_path, "lib_rec_cc.pyx")),
    # Optional/problematic ones (won't break the build if they fail):
    ("isp.cython_code.ccwt_cy",        os.path.join(cy_path, "ccwt_cy.pyx")),
    ("isp.cython_code.whiten",         os.path.join(cy_path, "whiten.pyx")),
]

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