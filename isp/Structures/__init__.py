import json
from dataclasses import fields, dataclass, asdict, is_dataclass
from enum import Enum, EnumMeta


class Cast:
    """
    Cast a value to the give class:
    """
    encoding = 'utf-8'

    def __new__(cls, value: any, _type: type) -> any:
        """
        Cast a value to the give class. This is useful to cast dict to dataclasses and some non-trivial types.
        This supports conversion to almost all python's primitive types + dataclasses and enum types.
        Examples:
                Cast('1', int) will cast '1' -> 1
                Cast({'name': 'Obama'}, MyDataclass) will cast dict -> MyDataclass
                Cast('run', MyEnum) will cast -> MyEnum
        :param value: Value to be cast into type.
        :param _type: The casting type, i.e: bool, int, flot, str, dataclass or enum.
        """
        _type = _type.mro()[0]
        value_type = type(value)
        if value_type == _type:
            # no need to cast.
            return value
        try:
            if _type == bool:
                value = json.loads(value.lower()) if isinstance(value, (str, bytes)) else bool(value)
            elif _type == str:
                value = str(value, Cast.encoding)
            elif issubclass(_type, (int, float)):
                value = _type(value) if not cls.__is_none(value) else _type(0)
            elif issubclass(_type, Enum):
                value = cls.cast_to_enum(value, _type)
            elif issubclass(_type, (list, dict)):
                value = json.loads(value) if isinstance(value, (str, bytes)) else _type(value)
            elif issubclass(_type, tuple):
                value = cls.cast_str_tuple(value) if isinstance(value, (str, bytes)) else _type(value)
            # convert to dataclass
            elif is_dataclass(_type):
                value = value if value_type != str else json.loads(value)
                value = cls.dataclass_from_dict(_type, value)
            else:
                print(f"{value} of type {value_type} can't be cast to {_type}")

        except Exception as e:
            print(e)
            raise ValueError(f"{value} can't be cats to {_type}")
        return value

    @staticmethod
    def __is_none(value: str):

        if value is None:
            return True
        elif type(value) == str and (value == '' or value.lower() == "null" or value.lower() == "none"):
            return True
        else:
            return False

    @classmethod
    def dataclass_from_dict(cls, kls: dataclass, dto: dict):

        if isinstance(dto, kls):
            # can't convert it already has the same type.
            # All good.
            return dto

        dto = cls.validate_dataclass(kls, dto)

        return kls(**dto)

    @classmethod
    def validate_dataclass(cls, kls: dataclass, dto: dict):
        """
        Forces the dictionary to have the same type of the parameter's declaration.
        If dictionary has None values the key is not used, instead dataclass will be constructed with default values.
        :param kls: Expect a dataclass type.
        :param dto: The dictionary to be validated.
        :return: A new dictionary that try to keep the same data type from dataclass.
        """
        valid_dic = {}
        fields_ = fields(kls)
        fields_list_lower = [f.name.lower() for f in fields(kls)]
        for k, value in dto.items():
            index = fields_list_lower.index(k.lower())  # Compare all in lower case. Avoid Caps sensitive.
            safe_key = fields_[index].name
            if not cls.__is_none(value):
                valid_dic[safe_key] = Cast(value, fields_[index].type)
        return valid_dic

    @staticmethod
    def cast_to_enum(value, enum_kls: EnumMeta):
        try:
            return enum_kls(value)
        except ValueError:
            return enum_kls[value]

    @staticmethod
    def cast_str_tuple(value):
        # convert to string in the case is bytes.
        value = Cast(value, str)
        # convert string tuple to list string the cas to list and get back to tuple.
        return tuple(Cast(value.replace('(', '[').replace(')', ']'), list))


@dataclass
class BaseDataClass:

    @classmethod
    def from_dict(cls, dto: dict):
        return Cast(dto, cls)

    def to_dict(self):
        return asdict(self)
