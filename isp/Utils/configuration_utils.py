#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
configuration_utils

"""

from configparser import ConfigParser


def parse_configuration_file(file_path: str) -> ConfigParser:
    """
    Read and parse a filename .ini or other valid extensions for ConfigParser

    Args:
        file_path: The full path of your configuration file

    Returns: An instance of ConfigParser
    """
    config = ConfigParser()
    config.read(file_path)
    return config