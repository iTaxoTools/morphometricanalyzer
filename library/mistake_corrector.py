from library.record import *
from typing import Set, List, Tuple, TextIO, Optional, Iterator


class HeaderFixer():
    """
    Class for applying fixes for the header of the data file.
    """

    required_fields = ['specimenid', 'species', 'sex', 'locality']

    def __init__(self, header: str):
        """
        Initialize with the header line of the data file.
        """
        self.fields: List[Optional[str]] = [field.strip().casefold()
                                            for field in header.split('\t')]
        self.field_names_corrections: List[Tuple[str, str]] = []
        self.metafields_ordered = True
        self.bad_metafields: List[str] = []
        self.nonessential_missing = False
        self.essential_missing = False
        self.non_unique_variables: List[str] = []
        self.correct_metafields()
        self.verify_metafields()
        self.check_missing_metafields()
        self.make_unique_variable_names()
        self.variables = set(self.fields) - set(HeaderFixer.required_fields)
        self.variables.discard(None)

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
        self.variables_num = len(self.fields) - metafield_len
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
        essential = {'specimenid', 'species'}
        nonessential = {'locality', 'sex'}
        if not nonessential <= set(self.fields):
            self.nonessential_missing = True
        if not essential <= set(self.fields):
            self.essential_missing = True
        return False

    def make_unique_variable_names(self) -> None:
        """
        Append running numbers to the repeated variable names.
        """
        for i, var in enumerate(self.fields):
            if var is None:
                continue
            repeated = self.fields[0:i].count(var)
            if repeated:
                self.fields[i] = var + str(repeated)
                self.non_unique_variables.append(var)

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
            print("Metadata columns are expected in the order specimenid, species, sex, locality. In the input file the order of these columns appears to be different. The program will proceed with the analyses, but please check if there may be a confusion affecting the metadata.", sep='', file=output_file)
        if self.essential_missing:
            print("The input file lacks at least one of the two columns, specimenid and species, that are required to run the analyses, and the program therefore could not be executed.", sep='', file=output_file)
        if self.nonessential_missing:
            print("The input file lacks at least one of the two columns, locality and sex, and certain analyses can therefore not be performed.", sep='', file=output_file)
        if self.bad_metafields:
            print("The following column(s) have been identified interspersed among the metedata columns: ",
                  ", ".join(self.bad_metafields),
                  ". These columns will not be considered to contain variables and will be ignored in all further analyses. If this is not intended, please change order of columns and repeat analysis.", sep='', file=output_file)
        if self.non_unique_variables:
            print("The following non-unique variable names were detected:",
                  ", ".join(self.non_unique_variables),
                  ". These have been renamed by appending numbers.", sep='', file=output_file)


class MistakeCorrector():
    """
    Iterates over the lines of a tab-separated data file and corrects certain mistakes.

    Yields corrected lines.
    """

    def __init__(self, lines: Iterator[str]):
        try:
            header = next(lines)
        except StopIteration:
            raise ValueError("The input file is empty")
        self.lines = lines
        self.header_fixer = HeaderFixer(header)
        self.at_header = True
        self.unique_specimenids: Set[str] = set()
        self.non_unique_specimenids: Set[str] = set()
        self.unusual_sexes: Set[str] = set()
        self.numbers_with_commas = False
        self.invalid_content_count = 0
        self.vars_with_numbers: Set[str] = set()

    def report(self, output_file: TextIO) -> None:
        self.header_fixer.report(output_file)
        if self.non_unique_specimenids:
            print("The column specimenid contains the following non-unique values which may indicate that measurements of one or several specimens will be entered twice in the analysis: ",
                  ", ".join(self.non_unique_specimenids), ".", sep='', file=output_file)
        if self.unusual_sexes:
            print("The column sex typically accepts as values only male, female, juvenile and larva, but it was found to also contain the following ones: ",
                  ", ".join(self.unusual_sexes), ".", sep='', file=output_file)
        if self.numbers_with_commas:
            print("Some or all variable fields contained values with commas; these have been converted to periods assuming that they were meant to be decimal separators as usual e.g. in Spanish, French or German.",
                  file=output_file)
        if self.invalid_content_count:
            print(f"A total of {self.invalid_content_count} cases with invalid content (not a number) were detected in the fields of one or several variables; this content has been deleted and in the analysis is treated as missing value.", file=output_file)
        empty_columns = self.header_fixer.variables - self.vars_with_numbers
        if empty_columns:
            for var in empty_columns:
                print(
                    f"Variable {var} does not appear to contain numerical values and is therefore excluded from all calculations.", file=output_file)

    def __iter__(self) -> "MistakeCorrector":
        return self

    def verify_unique_specimenid(self, record: Record) -> None:
        """
        Check that the record has a unique specimenid

        Writes errors into the Remark field
        """
        if record.specimenid in self.unique_specimenids:
            self.non_unique_specimenids.add(record.specimenid)
            record.remark_add("Non-unique specimenid value detected.")
        else:
            self.unique_specimenids.add(record.specimenid)

    def verify_sex(self, record: Record) -> None:
        """
        Corrects the sex attribute of the Record

        variants of male, female, juvenile and larva are corrected

        other values are noted
        """
        corrections = dict(
            male={"m", "m.", "mâle", "macho",
                  "Maennchen", "Männchen", "männlich"},
            female={"f", "f.", "fem", "fem.", "femelle", "hembra", "Weibchen"},
            juvenile={"j", "j.", "juv", "juv.", "juvenil"},
            larva={"l", "l.", "larv", "larv.", "larvae", "larve"}
        )
        sex = record.sex.casefold()
        for standard_sex, variants in corrections.items():
            if sex in variants:
                record.sex = standard_sex
                return
        if sex not in corrections:
            self.unusual_sexes.add(sex)
            record.remark_add(
                "The column sex contains a category different from male, female, juvenile or larva")
            return
        else:
            record.sex = sex

    def correct_values(self, record: Record) -> None:
        """
        Corrects the numerical values of variables

        Deletes non-numerical values
        """
        for field, value in record.variables.items():
            if ',' in value:
                value = value.replace(',', '.')
                self.numbers_with_commas = True
            try:
                value = str(float(value))
            except ValueError:
                value = ""
                self.invalid_content_count += 1
            else:
                self.vars_with_numbers.add(field)
            record.variables[field] = value

    def __next__(self) -> str:
        if self.at_header:
            self.at_header = False
            return self.yield_header()
        row = next(self.lines)
        record = Record(self.header_fixer.fields, row)
        self.verify_unique_specimenid(record)
        self.verify_sex(record)
        self.correct_values(record)
        return record.to_row()

    def yield_header(self) -> str:
        return "\t".join(field if field is not None else "" for field in self.header_fixer.fields) + "\t" + "Remark"
