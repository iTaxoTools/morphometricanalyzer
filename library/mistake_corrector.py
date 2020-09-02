from record import *
from typing import Set, List, Tuple, TextIO, Optional


class HeaderFixer():
    """
    Class for applying fixes for the header of the data file.
    """

    required_fields = ['specimenid', 'species', 'sex', 'locality']

    def __init__(self, header: str):
        """
        Initialize with the header line of the data file.
        """
        self.fields: List[Optional[str]] = [field.strip()
                                            for field in header.split('\t')]
        self.field_names_corrections: List[Tuple[str, str]] = []
        self.metafields_ordered = True
        self.bad_metafields: List[str] = []
        self.nonessential_missing = False
        self.essential_missing = False

    def correct_metafield(self, correct_name: str, variants: Set[str]) -> None:
        """
        if correct_name is in fields, do nothing
        otherwise replace first of the variants with the correct_name
        """
        if correct_name in self.fields:
            return
        for i, field in enumerate(self.fields):
            if field in variants:
                self.fields[i] = correct_name
                self.field_names_corrections.append((field, correct_name))
                break

    def correct_metafields(self) -> None:
        """
        Try to correct typos and synonyms of the metafields
        """
        metafields_variants: Dict[str, Set[str]] = dict(
            specimenid={'specimen id', 'specimen_id',
                        'specimen-id', 'specimen', 'specimen number'},
            species={'taxon'},
            sex=set(),
            locality={'localiti', 'locallity', 'site', 'location'}
        )
        for correct_name, variants in metafields_variants.items():
            self.correct_metafield(correct_name, variants)

    def verify_metafields(self) -> None:
        """
        Verify that metafields are in correct order and that their are no exraneous metafields.
        """
        # indices of metafields
        metafield_perm = [
            i for i, field in enumerate(self.fields) if field in HeaderFixer.required_fields]
        # check that metafields are properly ordered
        self.metafields_ordered = sorted(metafield_perm) == metafield_perm
        # calculate the length of the metafield portion of the header
        metafield_len = max(metafield_perm) + 1
        # collect the extraneous fields and delete them
        for i, field in enumerate(self.fields[0:metafield_len]):
            if field not in set(HeaderFixer.required_fields):
                self.fields[i] = None
                assert(field is not None)
                self.bad_metafields.append(field)

    def check_missing_metafields(self) -> bool:
        """
        Check if some metafields are missing,
        returns false if the essential metafields are missing
        """
        essential = {'specimen_id', 'species'}
        nonessential = {'locality', 'sex'}
        if not nonessential <= set(self.fields):
            self.nonessential_missing = True
        if not essential <= set(self.fields):
            self.essential_missing = True
        return False

    def report(self, output_file: TextIO) -> None:
        """
        Write the output of the correction into the output_file
        """
        # metadata fields names' corrections
        if self.field_names_corrections:
            print("Changes or corrections have been applied in the title of one or several metadata columns:",
                  *(", ".join([f"{given_name} was changed to {corrected_name}" for given_name,
                               corrected_name in self.field_names_corrections])),
                  file=output_file)
        if not self.metafields_ordered:
            print("Metadata columns are expected in the order specimenid, species, sex, locality. In the input file the order of these columns appears to be different. The program will proceed with the analyses, but please check if there may be a confusion affecting the metadata.")
        if self.essential_missing:
            print("The input file lacks at least one of the two columns, specimenid and species, that are required to run the analyses, and the program therefore could not be executed.")
        if self.nonessential_missing:
            print("The input file lacks at least one of the two columns, locality and sex, and certain analyses can therefore not be performed.")
        if self.bad_metafields:
            print("The following column(s) have been identified interspersed among the metedata columns: ",
                  ", ".join(self.bad_metafields),
                  ". These columns will not be considered to contain variables and will be ignored in all further analyses. If this is not intended, please change order of columns and repeat analysis.")

# class MistakeCorrector():
