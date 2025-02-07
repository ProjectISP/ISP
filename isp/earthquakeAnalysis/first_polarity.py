#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 20:26:28 2019

@author: robertocabieces
"""

import shutil
import subprocess
import pandas as pd
import os
from obspy import read_events, Catalog
from obspy.core.event import Origin
from isp import FOC_MEC_PATH, FOC_MEC_BASH_PATH, ROOT_DIR
from isp.Utils import ObspyUtil
from isp.earthquakeAnalysis import focmecobspy
from isp.Utils.subprocess_utils import exc_cmd
import re

class FirstPolarity:

    def __init__(self):
        """
        Manage FOCMEC files for run nll program.

        Important: The  obs_file_path is provide by the class :class:`PickerManager`.

        :param obs_file_path: The file path of pick observations.
        """

    @staticmethod
    def check_no_empty(file_path):
        count = 0
        with open(file_path, 'r') as file:
            for line in file:
                if line.strip():  # Check if the line contains characters (ignoring whitespace)
                    count += 1
                if count > 4:
                    return True
        return False
    @staticmethod
    def __validate_dir(dir_path):
        if not os.path.isdir(dir_path):
            raise FileNotFoundError("The dir {} doesn't exist.".format(dir_path))

    @property
    def root_path(self):
        root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "location_output")
        self.__validate_dir(root_path)
        return root_path

    @property
    def get_loc_dir(self):
        loc_dir = os.path.join(self.root_path, "loc")
        self.__validate_dir(loc_dir)
        return loc_dir

    @property
    def get_foc_dir(self):
        first_polarity_dir = os.path.join(self.root_path, "first_polarity")
        self.__validate_dir(first_polarity_dir)
        return first_polarity_dir

    def get_dataframe(self, location_file):
        Station = []
        Az = []
        Dip = []
        Motion = []
        df = pd.read_csv(location_file, delim_whitespace=True, skiprows=17)
        for i in range(len(df)):
            if df.iloc[i].RAz > 0:
                sta = str(df.iloc[i].PHASE)
                if len(sta) >= 5:
                    sta = sta[0:4]
                az = df.iloc[i].SAzim
                dip = df.iloc[i].RAz
                m = df.iloc[i].Pha
                ph = str(df.iloc[i].On)

                if dip >= 90:
                    dip = 180 - dip

                if ph[0] == "P" and m != "?":
                    Az.append(az)
                    Dip.append(dip)
                    Motion.append(m)
                    Station.append(sta)

                if ph[0] == "S" and m != "?":
                    Az.append(az)
                    Dip.append(dip)
                    Motion.append(m)
                    Station.append(sta)

        return Station, Az, Dip, Motion

    def get_NLL_info(self):
        location_file = os.path.join(self.get_loc_dir, "last.hyp")
        if os.path.isfile(location_file):
            cat = read_events(location_file)
            event = cat[0]
            origin = event.origins[0]
            return origin
        else:
            raise FileNotFoundError("The file {} doesn't exist. Please, run location".format(location_file))

    def create_input(self, file_last_hyp, header):

        Station, Az, Dip, Motion = self.get_dataframe(file_last_hyp)

        one_level_up = os.path.dirname(file_last_hyp)
        two_levels_up = os.path.dirname(one_level_up)
        dir_path = os.path.join(two_levels_up, "first_polarity")

        if os.path.isdir(dir_path):
            self.clean_output_folder_focmec(dir_path)
        else:

            os.makedirs(dir_path)

        temp_file = os.path.join(dir_path, "test.inp")
        N = len(Station)

        with open(temp_file, 'wt') as f:
            #f.write("\n")  # first line should be skipped!
            f.write(header)
            f.write("\n")
            for j in range(N):
                f.write("{:4s}  {:6.2f}  {:6.2f}{:1s}\n".format(Station[j], Az[j], Dip[j], Motion[j]))

        # Now move the template running to the first_polarity folder
        shutil.copy(FOC_MEC_BASH_PATH, dir_path)

        return temp_file

    def clean_output_folder_focmec(self, dir_path):

        """
        Cleans the destination folder and creates symbolic links for all files in the source folder.

        Args:
        destination_folder (str): Path to the destination folder.
        source_folder (str): Path to the source folder.

        """

        # Clean the destination folder
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)  # Remove files or symbolic links
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)  # Remove

    def run_focmec_csh(self):
         #old_version, need bash or csh
         command=os.path.join(self.get_foc_dir, 'rfocmec_UW')
         exc_cmd(command)

    def run_focmec(self, input_focmec_path, num_wrong_polatities):

        # binary file
        command = os.path.join(FOC_MEC_PATH, 'focmec')
        dir_name = os.path.dirname(input_focmec_path)

        # first_polarity/output
        output_path = os.path.join(dir_name, "output")

        # first_polarity path
        focmec_bash_path = os.path.join(os.path.dirname(input_focmec_path), "focmec_run")

        if os.path.isdir(output_path):
            pass
        else:
            os.makedirs(output_path)

        # edit number of accepted wrong polarities
        self.edit_focmec_run(num_wrong_polatities, focmec_bash_path)

        shutil.copy(input_focmec_path, '.')
        with open(focmec_bash_path, 'r') as f, open('./log.txt', 'w') as log:
            p = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=log)
            f = open(focmec_bash_path, 'r')
            string = f.read()
            # This action has created focmec.lst at ./isp
            out, errs = p.communicate(input=string.encode(), timeout=2.0)
            # This action has created mechanism.out at ./isp
        #time.sleep(5)
        exec_path = os.path.dirname(ROOT_DIR)
        mechanism_out = os.path.join(exec_path, "mechanism.out")
        focmec_lst = os.path.join(exec_path, "focmec.lst")
        log_file = os.path.join(exec_path, "log.txt")
        test_inp = os.path.join(exec_path, "test.inp")
        output_ref = FirstPolarity.extract_name(focmec_lst)
        # Sanitize file name
        file_output_name = (f"{output_ref['date'].replace('/', '-')}_"
                            f"{output_ref['time'].replace(':', '_')}")

        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)

        # Move and rename files
        if os.path.exists(mechanism_out):
            shutil.move(mechanism_out, os.path.join(output_path, file_output_name + '.out'))
        if os.path.exists(focmec_lst):
            shutil.move(focmec_lst, os.path.join(output_path, file_output_name + '.lst'))
        if os.path.exists(log_file):
            shutil.move(log_file, os.path.join(output_path, file_output_name + '.txt'))
        if os.path.exists(test_inp):
            shutil.move(test_inp, os.path.join(output_path, file_output_name + '.inp'))

    def extract_focmec_info(self, focmec_path):
        catalog: Catalog = focmecobspy._read_focmec(focmec_path)
        # TODO Change to read_events in new version of ObsPy >= 1.2.0
        #catalog = read_events(os.path.join(self.get_foc_dir, 'focmec.lst'),format="FOCMEC")
        #plane_a = catalog[0].focal_mechanisms[0].nodal_planes.nodal_plane_1
        focal_mechanism = self.__get_minimum_misfit(catalog[0].focal_mechanisms)
        return catalog, focal_mechanism

    def __get_minimum_misfit(self, focal_mechanism):
        mismifits = []
        for i, focal in enumerate(focal_mechanism):
            if focal.misfit is not None:
                mismifits.append(focal.misfit)
        if len(mismifits)>0:
            index = mismifits.index(min(mismifits))
            return focal_mechanism[index]
        else:
            return None

    def edit_focmec_run(self, new_float_value, focmec_bash_path):
        """
        Edits a specific line in the text file to update the float value and writes it to a new location.

        Parameters:
            input_path (str): The path to the input template file.
            output_path (str): The path to save the modified file.
            new_float_value (float): The new float value to replace in the specific line.
        """

        #output_path = os.path.join(os.path.dirname(FOC_MEC_BASH_PATH), "focmec_run")
        # Read the file and store its content
        with open(focmec_bash_path, 'r') as file:
            lines = file.readlines()

        # Edit the specific line containing "allowed P polarity erors..[0]"
        for i, line in enumerate(lines):
            if "allowed P polarity erors" in line:
                # Split the line and replace the float value at the beginning
                parts = line.split()
                parts[0] = f"{new_float_value:.1f}"  # Format the float with one decimal place
                lines[i] = " ".join(parts) + '\n'  # Reconstruct the line
                break

        # Write the modified content to the new file
        with open(focmec_bash_path, 'w') as file:
            file.writelines(lines)

    def parse_solution_block(self, solution_text):
        """
        Parses a block of text containing Dip, Strike, Rake and other information
        and returns a dictionary with structured data.

        Parameters:
            solution_text (str): Text block containing solution information.

        Returns:
            dict: Parsed solution data.
        """
        parsed_data = {}

        # Patterns for parsing
        dip_strike_rake_pattern = r"Dip,Strike,Rake\s+([\d\.\-]+)\s+([\d\.\-]+)\s+([\d\.\-]+)"
        auxiliary_pattern = r"Dip,Strike,Rake\s+([\d\.\-]+)\s+([\d\.\-]+)\s+([\d\.\-]+)\s+Auxiliary Plane"
        lower_hem_pattern = r"Lower Hem\. Trend, Plunge of ([A-Z]),[N|T]\s+([\d\.\-]+)\s+([\d\.\-]+)\s+([\d\.\-]+)\s+([\d\.\-]+)"
        b_axis_pattern = r"B trend, B plunge, Angle:\s+([\d\.\-]+)\s+([\d\.\-]+)\s+([\d\.\-]+)"
        polarity_pattern = r"P Polarity error at\s+([A-Z]+)"
        weight_pattern = r"P Polarity weights:\s+([\d\.\-]+)"
        total_weight_pattern = r"Total P polarity weight is\s+([\d\.\-]+)"
        total_number_pattern = r"Total number:\s+(\d+)"

        # Main Plane
        main_match = re.search(dip_strike_rake_pattern, solution_text)
        if main_match:
            parsed_data['Main Plane'] = {
                'Dip': float(main_match.group(1)),
                'Strike': float(main_match.group(2)),
                'Rake': float(main_match.group(3)),
            }

        # Auxiliary Plane
        aux_match = re.search(auxiliary_pattern, solution_text)
        if aux_match:
            parsed_data['Auxiliary Plane'] = {
                'Dip': float(aux_match.group(1)),
                'Strike': float(aux_match.group(2)),
                'Rake': float(aux_match.group(3)),
            }

        # Lower Hemisphere Trends and Plunges
        for match in re.finditer(lower_hem_pattern, solution_text):
            plane_key = match.group(1)  # 'A' or 'P'
            parsed_data[f'{plane_key},T'] = {
                'Trend': float(match.group(2)),
                'Plunge': float(match.group(3)),
            }
            parsed_data[f'{plane_key},N'] = {
                'Trend': float(match.group(4)),
                'Plunge': float(match.group(5)),
            }

        # B Axis
        b_match = re.search(b_axis_pattern, solution_text)
        if b_match:
            parsed_data['B Axis'] = {
                'Trend': float(b_match.group(1)),
                'Plunge': float(b_match.group(2)),
                'Angle': float(b_match.group(3)),
            }

        # Polarity Error and Weights
        polarity_match = re.search(polarity_pattern, solution_text)
        if polarity_match:
            parsed_data['P Polarity Error'] = polarity_match.group(1)

        weight_match = re.search(weight_pattern, solution_text)
        if weight_match:
            parsed_data['P Polarity Weights'] = float(weight_match.group(1))

        total_weight_match = re.search(total_weight_pattern, solution_text)
        if total_weight_match:
            parsed_data['Total P Polarity Weight'] = float(total_weight_match.group(1))

        total_number_match = re.search(total_number_pattern, solution_text)
        if total_number_match:
            parsed_data['Total Number'] = int(total_number_match.group(1))

        return parsed_data

    @staticmethod
    def set_head(event_file):
        header = "\n"
        # helps to set a string in comments of focmec input
        try:
            origin: Origin = ObspyUtil.reads_hyp_to_origin(event_file)
            origin_time_formatted_string = origin.time.datetime.strftime("%d/%m/%Y, %H:%M:%S.%f")

            lat = str(origin.longitude)
            lon = str(origin.latitude)
            depth = str(origin.depth / 1000)
            header = origin_time_formatted_string + " " + lat + " " + lon + " " + depth
        except:
            pass
        return header

    @staticmethod
    def extract_name(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        for i, line in enumerate(lines):
            if "Input" in line:
                # Extract the line after "Input"
                epicenter_line = lines[i + 1].strip()
                # Split the line into date, time, and coordinates
                parts = epicenter_line.split(',')
                if len(parts) < 2:
                    continue  # Skip if line format is unexpected
                date = parts[0].strip()
                time_and_coordinates = parts[1].strip().split()
                time = time_and_coordinates[0]
                longitude = float(time_and_coordinates[1])
                latitude = float(time_and_coordinates[2])
                depth_km = float(time_and_coordinates[3])
                # Store data in a dictionary
                return {
                    "date": date,
                    "time": time,
                    "longitude": longitude,
                    "latitude": latitude,
                    "depth_km": depth_km}

        return None  # Return None if "Input" is not found

    @staticmethod
    def find_files(base_path):
        pattern = re.compile(r'.*lst$')  # Match files ending with ".grid0.loc.hyp"
        obsfiles = []  # Initialize the list
        path_to_find = os.path.join(base_path, 'first_polarity/output')

        for top_dir, _, files in os.walk(path_to_find):
            for file in files:
                # If the file matches the desired pattern, add it to the list
                if pattern.match(file):
                    obsfiles.append(os.path.join(top_dir, file))

        return obsfiles
    @staticmethod
    def extract_station_data(file_path):
        stations = []
        azimuths = []
        dips = []
        motions = []

        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Find the start of the station data
        station_data_started = False
        for line in lines:
            # Skip lines until the header is detected
            if line.strip().startswith("Statn"):
                station_data_started = True
                continue

            # Skip processing lines until after header
            if not station_data_started:
                continue

            # Stop parsing if the data block ends
            if line.strip().startswith("Polarities/Errors:"):
                break

            # Process station data lines
            parts = line.split()
            if len(parts) < 4:
                continue  # Skip malformed lines

            try:
                # Extract fields
                station = parts[0]
                azimuth = float(parts[1])
                dip = float(parts[2])
                motion = parts[3]

                # Skip rows with '?' in motion
                if '?' in motion:
                    continue

                # Append to respective lists
                stations.append(station)
                azimuths.append(azimuth)
                dips.append(dip)
                motions.append(motion)

            except ValueError:
                # Ignore lines with invalid numeric values
                continue

        return stations, azimuths, dips, motions

    @staticmethod
    def find_loc_mec_file(loc_file):

        """
        Parameters
        ----------
        loc_file: Full path hyp file
        foc_mec_basedir: Full path focmec_output
        Returns
        -------

        """
        file_found = None
        origin: Origin = ObspyUtil.reads_hyp_to_origin(loc_file)
        time = origin.time.datetime.strftime("%d-%m-%Y_%H_%M_%S.%f")
        expected_foc_mec = time + ".lst"

        pattern = re.compile(r'.*lst$')  # Match files ending with ".grid0.loc.hyp"
        obsfiles = []  # Initialize the list

        foc_mec_basedir = os.path.dirname(loc_file)
        foc_mec_basedir = os.path.dirname(foc_mec_basedir)
        path_to_find = os.path.join(foc_mec_basedir, 'first_polarity/output')

        for top_dir, _, files in os.walk(path_to_find):
            for file in files:
                # If the file matches the desired pattern, add it to the list
                if pattern.match(file):
                    obsfiles.append(file)

        # obsfiles -->  list of lst files
        # expected_foc_mec --> expected file extracted from hyp file

        for file in obsfiles:
            if file == expected_foc_mec:
                file_found = file

        return file_found




