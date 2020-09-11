from typing import Iterator, List, Set, TextIO

import pandas as pd


def mean_and_others(col: pd.Series) -> str:
    num = col.count()
    if num < 2:
        return f"{col.mean():.3f}, N = {num}"
    else:
        return f"{col.mean():.3f} ± {col.std():.3f} ({col.min():.3f} - {col.max():.3f}), N = {num}"


def mean_analysis(table: pd.core.groupby.GroupBy, variables: Set[str]) -> Iterator[pd.DataFrame]:
    # create new column names
    meanvar_rename = {var: f"Mean{var.capitalize()}" for var in variables}

    # table of means
    yield table.mean().rename(columns=meanvar_rename)
    # table of means, stds, mins, maxes and counts
    yield table.aggregate(mean_and_others).rename(columns=meanvar_rename)


def analyse(buf: TextIO, output_file: TextIO, variables: Set[str], analyses: List[List[str]]) -> None:
    table = pd.read_table(buf, usecols=(
        ['specimenid', 'species', 'sex', 'locality'] + list(variables)))
    for analysis in analyses:
        do_analysis(table, variables, analysis, output_file)
        output_file.write("\n")


def do_analysis(table: pd.DataFrame, variables: Set[str], analysis: List[str], output_file: TextIO) -> None:

    # groupby doesn't behave as needed if analysis is empty
    groupedtable = table.groupby(analysis) if analysis else table

    for table in mean_analysis(groupedtable, variables):
        table.to_csv(output_file, float_format="%.3f", sep='\t')
        output_file.write("\n")
