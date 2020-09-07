from typing import TextIO, Set
import pandas as pd


def mean_and_others(col: pd.Series) -> str:
    num = col.count()
    if num < 2:
        return f"{col.mean():.3f}, N = {num}"
    else:
        return f"{col.mean():.3f} ± {col.std():.3f} ({col.min():.3f} - {col.max():.3f}), N = {num}"


def analyse(buf: TextIO, output_file: TextIO, variables: Set[str]) -> None:
    table = pd.read_table(buf, usecols=(
        ['specimenid', 'species', 'sex', 'locality'] + list(variables)))
    meanvar_rename = {var: f"Mean{var.capitalize()}" for var in variables}
    meantable = table.groupby('species').mean().rename(columns=meanvar_rename)
    meantable.to_csv(output_file, float_format="%.3f", sep='\t')
    mean_and_others_table = table.groupby('species').aggregate(
        mean_and_others).rename(columns=meanvar_rename)
    mean_and_others_table.to_csv(output_file, float_format="%.3f", sep='\t')
