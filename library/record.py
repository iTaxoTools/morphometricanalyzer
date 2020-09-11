#!/bin/env python3
from typing import List, Dict, Optional


class Record():
    """
    Represents a row of a data file.

    Always has keys: specimenid, species, locality, sex

    variables attribute contains the variables in the row, excluding the metadata
    """

    metadata_fields = {'specimenid', 'species', 'sex', 'locality', 'remark'}

    def __init__(self, fields: List[Optional[str]], row: str):
        """
        Initializes a record with fields from fields and value from row.
        row is a line of tab-separated data file.
        """
        # subdictionary that contains only the variables
        self.fields = fields
        self.variables: Dict[str, str] = {}
        self.specimenid = ""
        self.species = ""
        self.sex = ""
        self.locality = ""
        self.remark = ""
        for field, value in zip(fields, row.split('\t')):
            value = value.strip()  # remove the surrounding spaces
            if field is None:
                continue
            elif field in Record.metadata_fields:
                # metadata fields
                self.__setattr__(field, value)
            else:
                # variable fields
                self.variables[field] = value

    def __getitem__(self, key: str) -> str:
        if key in Record.metadata_fields:
            try:
                return self.__getattribute__(key)
            except AttributeError as ex:
                raise KeyError from ex
        else:
            return self.variables[key]

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Same as dict.get
        """
        if key in Record.metadata_fields:
            try:
                return self.__getattribute__(key)
            except AttributeError as ex:
                return default
        else:
            return self.variables.get(key, default)

    def __setitem__(self, key: str, value: str) -> None:
        if key in Record.metadata_fields:
            self.__setattr__(key, value)
        else:
            self.variables[key] = value

    def to_row(self) -> str:
        """
        Prints the record as a row of a tab-separated data file
        """
        return "\t".join(self[field] if field is not None else "" for field in self.fields)

    def remark_add(self, remark: str) -> None:
        """
        Adds remark into self.remark, adding semicolon if required
        """
        if self.remark:
            self.remark += f"; {remark}"
        else:
            self.remark = remark
