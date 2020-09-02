#!/bin/env python3
from typing import List, Dict


class Record():
    """
    Represents a row of a data file.

    Always has keys: specimenid, species, locality, sex

    variables attribute contains the variables in the row, excluding the metadata
    """

    metadata_fields = {'specimenid', 'species', 'locality', 'sex'}

    def __init__(self, fields: List[str], row: str):
        """
        Initializes a record with fields from fields and value from row.
        row is a line of tab-separated data file.
        """
        # subdictionary that contains only the variables
        self.variables: Dict[str, str] = {}
        for field, value in zip(fields, row.split('\t')):
            value = value.strip()  # remove the surrounding spaces
            if field in Record.metadata_fields:
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

    def __setitem__(self, key: str, value: str) -> None:
        if key in Record.metadata_fields:
            self.__setattr__(key, value)
        else:
            self.variables[key] = value
