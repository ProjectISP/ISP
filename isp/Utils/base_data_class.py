#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
base_data_class
"""

from dataclasses import dataclass, asdict
from isp.Structures import Cast

@dataclass
class BaseDataClass:

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, dto: dict):
        return Cast(dto, cls)