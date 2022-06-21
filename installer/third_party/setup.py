#!/bin/sh

# ******************************************************************************
# *   ISP: Integrated Seismic Program                                          *
# *                                                                            *
# *   A Python GUI for earthquake seismology and seismic signal processing     *
# *                                                                            *
# *   Copyright (C) 2022     Roberto Cabieces                                  *
# *   Copyright (C) 2022     Thiago C. Junqueira                               *
# *   Copyright (C) 2022     Jesús Relinque                                    *
# *   Copyright (C) 2022     Ángel Vera                                        *
# *                                                                            *
# *   This file is part of ISP.                                                *
# *                                                                            *
# *   ISP is free software: you can redistribute it and/or modify it under the *
# *   terms of the GNU Lesser General Public License (LGPL) as published by    *
# *   the Free Software Foundation, either version 3 of the License, or (at    *
# *   your option) any later version.                                          *
# *                                                                            *
# *   ISP is distributed in the hope that it will be useful for any user or,   *
# *   institution, but WITHOUT ANY WARRANTY; without even the implied warranty *
# *   of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU      *
# *   Lesser General Public License (LGPL) for more details.                   *
# *                                                                            *
# *   You should have received a copy of the GNU LGPL license along with       *
# *   ISP. If not, see <http://www.gnu.org/licenses/>.                         *
# ******************************************************************************

import glob
import os
import shutil
import sys
import subprocess as sb
from distutils.core import setup
import distutils.command.build
from setuptools.command.build_ext import build_ext

# Get the current dir.
CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

# Get the build directory. 
##############################################
# THIS IS A SPANISH GUARRADA. 
# El instalador es un poco ñaposo y tenía las
# rutas hardcodeadas, cosa que no puede ser.
# No me ha quedado más remedio que poner esta
# guarrada para poder coger la ruta final de
# instalacion a lo bestia, ya que el instalador
# tal y como está programado, no la tiene en
# en cuenta. Sería mucho más costoso rehacer
# el instalador en condiciones, por lo que,
# de momento, creo que puede quedarse así.
##############################################
BUILD_BASE = CURRENT_PATH
build_base_long = [arg[12:].strip("= ") for arg in sys.argv if arg.startswith("--build-lib")]
build_base_short = [arg[2:].strip(" ") for arg in sys.argv if arg.startswith("-b")]
build_base_arg = build_base_long or build_base_short
if build_base_arg:
    BUILD_BASE = os.path.abspath(build_base_arg[0])

   
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
            std_out, std_err = p.communicate(timeout=20)
        except sb.TimeoutExpired:
            p.kill()
            std_out, std_err = p.communicate()  # try again if timeout fails.
        if p.returncode != 0:  # Bad error.
            raise sb.CalledProcessError(p.returncode, std_err.strip())
        elif len(std_err) != 0:  # Some possible errors trowed by the running subprocess, but not critical.
            raise sb.SubprocessError(std_err.strip())
        return std_out.strip()


def extract_nll():
    extract_at = os.path.join(CURRENT_PATH, "NLL7")
    try:
        if not os.path.isdir(extract_at):
            os.makedirs(extract_at)
        else:
            shutil.rmtree(extract_at)
            os.makedirs(extract_at)
        shutil.unpack_archive(CURRENT_PATH + "/NLL7.00_src.tgz", extract_at)
        os.makedirs(os.path.join(extract_at, "bin"))
    except IOError as error:
        print(error)
    return extract_at


def extract_focmec():
    extract_at = os.path.join(CURRENT_PATH, "FOCMEC")
    try:
        if not os.path.isdir(extract_at):
            os.makedirs(extract_at)
        else:
            shutil.rmtree(extract_at)
            os.makedirs(extract_at)
        shutil.unpack_archive(CURRENT_PATH + "/focmec.tar", extract_at)
    except IOError as error:
        print(error)
    return extract_at


def extract_mti():
    extract_at = os.path.join(CURRENT_PATH, "MTI","bin")
    
    try:            
        if not os.path.isdir(extract_at):
            os.makedirs(extract_at)
        else:
            shutil.rmtree(extract_at)
            os.makedirs(extract_at)
        shutil.unpack_archive(CURRENT_PATH + "/mti.tar.gz", extract_at)
    except IOError as error:
        print(error)
    return extract_at


class CustomBuildExtCommand(build_ext):

    def run(self):
        print("Extracting files...")
        nll_dir = extract_nll()
        focmec_dir = extract_focmec()
        mti_dir = extract_mti()
        print("Extraction finished.")

        print("Compiling...")
        self.make_nll(nll_dir)
        self.make_focmec(focmec_dir)
        self.make_mti(mti_dir)
        print("Compilation finished.")
        
        # run build
        build_ext.run(self)


    def make_nll(self, nll_dir):
        src_path = os.path.join(nll_dir, "src")
        bin_path = os.path.join(nll_dir, "bin")
        build_dir = BUILD_BASE + '/NLL7'
        command = "export MYBIN={bin_path}; cd {src_path}; make -R all ".format(bin_path=bin_path, src_path=src_path)
        try:
            exc_cmd(command, shell=True)
            print("NLL7 successfully compiled.")
        except sb.CalledProcessError as e:  # this is a bad error.
            print("Error on trying to run NLL7 make file.")
            print(e)
        except sb.SubprocessError:  # some warnings nothing so bad.
            print("NLL7 compiled with warnings.")
              
        # Deploying.   
        print("Deploying NLL7 into < " + build_dir + "> folder...")
        if not os.path.isdir(build_dir):
            os.makedirs(build_dir)
        shutil.move(bin_path, build_dir)
        
        # Remove the extract folder.
        print("Removing the NLL7 extract folder...")
        shutil.rmtree(CURRENT_PATH+'/NLL7')


    def make_focmec(self, focmec_dir):
        src_path = os.path.join(focmec_dir, "src")
        command = "sh build_package_sh"
        build_dir = BUILD_BASE + '/FOCMEC'
        try:
            exc_cmd(command, cwd=src_path, shell=True)
            print("FOCMEC successfully compiled.")
        except sb.CalledProcessError as e:  # this is a bad error.
            print("Error on trying to run focmec make file.")
            print(e)
        except sb.SubprocessError:  # some warnings nothing so bad.
            print("FOCMEC compiled with warnings.")
             
        # Deploying.   
        print("Deploying FOCMEC into < " + build_dir + "> folder...")
        if not os.path.isdir(build_dir):
            os.makedirs(build_dir)
        shutil.move(focmec_dir+'/bin', build_dir)
        
        # Remove the extract folder.
        print("Removing the FOCMEC extract folder...")
        shutil.rmtree(CURRENT_PATH+'/FOCMEC')


    def make_mti(self, mti_dir):
        src_path = os.path.join(mti_dir)
        command = "sh compile_mti.sh"
        build_dir = BUILD_BASE + '/MTI'
        try:
            exc_cmd(command, cwd=src_path, shell=True)
            print("MTI successfully compiled.")
        except sb.CalledProcessError as e:  # this is a bad error.
            installed = False
            print("Error on trying to run MTI make file.")
            print(e)
            return
        except sb.SubprocessError:  # some warnings nothing so bad.
            print("MTI compiled with warnings.")
            
        # Cleaning the folders.
        print("Cleaning MTI folders...")
        files = glob.glob(src_path+'/*')
        for item in files:
            if item.endswith(".o") or item.endswith(".inc")   \
              or item.endswith(".sh") or item.endswith(".for"):
                os.remove(os.path.join(src_path, item))
             
        # Deploying.   
        print("Deploying MTI into < " + build_dir + "> folder...")
        if not os.path.isdir(build_dir):
            os.makedirs(build_dir)
        shutil.move(src_path, build_dir)
        
        # Remove the extract folder.
        print("Removing the MTI extract folder...")
        shutil.rmtree(CURRENT_PATH+'/MTI')


# Setup.
setup(cmdclass={'ISP_ThirdParty': CustomBuildExtCommand,},py_modules=[])

