"""
dataclass style mechanics to give classes and instances the ability to iterate over a subset
of attributes for serialization operations like storing/restoring state provide 'as_dict' behavior and so.
Contrary to dataclass (which inspires anyway the mechanic) this allows to mark only  some specific attributes
for the behavior so giving more control over typical dataclass features.


example:

class BetterDataClass(DataAttrClass):
    field1: DataAttr[str, "None"]
    a: DataAttr[int| None] = 4
    b: DataAttr[int| None] = None

attributes defined so, looks like class attributes but will then be automatically initialized in the instance
with defaults (if set) from the class attribute definition.
Type checkers should treat this as if they were 'normal' type hints i.e.

class BetterDataClass:
    field1: str
    a: int | None = 4
    b: int | None = None

in the meantime the base DataAttrClass will take care of setting these on the instance so that they become
effective instance members. If no default has been set in the class attribute definition then no attribute will
be created for the instance (this has to do with slots automatic definitions once the mechanic is in place)

DataAttr type hint leverages typing.Annotated to achieve this

"""

import enum
import typing

from . import datetime_from_epoch

if typing.TYPE_CHECKING:
    from typing import Any, ClassVar, Final, Mapping


class _DataAttrParam(type):
    def __new__(cls, name: str):
        return type(name, (), {})


class DataAttrParam(_DataAttrParam, enum.Enum):
    no_slot = "no_slot"
    hide = "hide"
    stored = "stored"


class _DataAttrDef(typing.NamedTuple):
    annotation: type
    params: typing.Iterable[DataAttrParam]
    slot: bool
    show: bool
    stored: bool


type DataAttr[TypeHint, *params] = typing.Annotated[TypeHint, *params]

type timestamp_f = float
type timestamp_i = int


class DataAttrClass(typing.Generic[typing.TypeVar("DataAttr", default=object)]):
    # Inheriting from Generic is needed in our hierarchy to correctly support MRO
    # The Generic type is defaulted to object so that we don't have to bother in our
    # eventual Generic implementations along the hierarchy.
    # That's a bit of overhead for nothing but as for now that's it

    DataAttr = DataAttr
    DataAttrParam = DataAttrParam

    if typing.TYPE_CHECKING:
        type _DataAttrsT = Mapping[str, _DataAttrDef]
        __data_attrs: Final[_DataAttrsT]
        """DataAttr attributes defined (locally) in the class."""
        _DATA_ATTRS: Final[_DataAttrsT]
        """DataAttr attributes defined in the class and ancestors (i.e. all DataAttrs)"""

    _DATA_ATTRS = {}

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cls = self.__class__
        for attr in cls._DATA_ATTRS:
            try:
                setattr(self, attr, getattr(cls, attr))
            except AttributeError:
                pass  # no default in class definition

    def __init_subclass__(cls, *args, **kwargs):

        # type hints defined as DataAttr will have these properties:
        # typehint.__name__ == aliasname (i.e. DataAttr)
        # typehint.__value__ == underlying typehint i.e. typing.Annotated
        # typehint.__value__.__metadata__ == tuple of args to Annotated

        dataattr_ann = {
            name: typehint
            for name, typehint in cls.__annotations__.items()
            if typehint.__name__ is "DataAttr"
        }

        # Check if this level of the hierarchy defines any new dataattr
        # or it just (eventually) inherit the bases _DATA_ATTRS
        if dataattr_ann:

            def _build_dataattr_def(typehint):
                _, *params = (
                    typehint.__args__
                )  # skip first Annotated param (underlying type)
                slot = DataAttrParam.no_slot not in params
                show = DataAttrParam.hide not in params
                stored = DataAttrParam.stored in params
                return _DataAttrDef(typehint, params, slot, show, stored)

            data_attrs: "DataAttrClass._DataAttrsT" = {
                name: _build_dataattr_def(typehint)
                for name, typehint in dataattr_ann.items()
            }
            # Now 'enrich' the current class _DATA_ATTRS
            cls.__data_attrs = data_attrs # type: ignore

            cls._DATA_ATTRS = {} # type: ignore
            for base in reversed(cls.mro()):
                try:
                    cls._DATA_ATTRS |= base.__data_attrs # type: ignore
                except AttributeError:
                    pass
            # eventually add slots
            slots = tuple(
                name for name, dataattr_def in data_attrs.items() if dataattr_def.slot
            )
            cls.__slots__ = cls.__slots__ + slots

        return super().__init_subclass__(*args, **kwargs)

    def as_dict(self):
        """Returns a dict with all of the (raw) attrs typed as DataAttr. Only exclude those
        marked by 'hide/no_show'."""
        return {
            attr: getattr(self, attr)
            for attr, dataattr_def in self.__class__._DATA_ATTRS.items()
            if dataattr_def.show
        }

    def as_formatted_dict(self):
        """Returns a dict with all of the attrs typed as DataAttr. Similar to 'as_dict'
        this tries to 'pretty' format data in the dict useful for simple presentations.
        Only exclude those marked by 'hide/no_show'."""
        _format_attr = self._format_attr
        result = {
            attr: _format_attr(attr)
            for attr, dataattr_def in self.__class__._DATA_ATTRS.items()
            if dataattr_def.show
        }
        return result

    def _format_attr(self, attr: str, /):
        value = getattr(self, attr)
        _type = type(value)
        if _type in (timestamp_f, timestamp_i):
            return datetime_from_epoch(value).isoformat()
        elif _type == float:
            return round(value, 2)
        return value

    def restore(self, data: "Mapping[str, Any]", /):
        for attr, dataattr_def in self.__class__._DATA_ATTRS.items():
            if dataattr_def.stored:
                try:
                    setattr(self, attr, data[attr])
                except KeyError:
                    pass

    def store(self) -> "Mapping[str, Any]":
        return {
            attr: getattr(self, attr)
            for attr, dataattr_def in self.__class__._DATA_ATTRS.items()
            if dataattr_def.stored
        }
