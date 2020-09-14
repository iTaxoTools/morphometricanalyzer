from typing import Iterator, List, Set, TextIO

import pandas as pd
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.base import HolderTuple


def mean_and_others(col: pd.Series) -> str:
    """
    Returns a string that contains the mean and related values in a formated way
    """
    num = col.count()
    if num == 0:
        return ""
    elif num < 2:
        return f"{col.mean():.3f}, N = {num}"
    else:
        return f"{col.mean():.3f} ± {col.std():.3f} ({col.min():.3f} - {col.max():.3f}), N = {num}"


def median_and_others(col: pd.Series) -> str:
    """
    Returns a string that contains the median and related values in a formated way
    """
    num = col.count()
    if num == 0:
        return ""
    elif num < 2:
        return f"{col.median():.3f}, N = {num}"
    else:
        return f"{col.median():.3f}, {col.quantile(0.75):.3f} - {col.quantile(0.25):.3f} ({col.min():.3f} - {col.max():.3f}), N = {num}"


def mean_analysis(table: pd.core.groupby.GroupBy, variables: Set[str]) -> Iterator[pd.DataFrame]:
    """
    Returns two tables, one with means and another with means and other values
    """
    # create new column names
    meanvar_rename = {var: f"Mean{var.capitalize()}" for var in variables}

    # table of means
    yield table.mean().rename(columns=meanvar_rename)
    # table of means, stds, mins, maxes and counts
    yield table.aggregate(mean_and_others).rename(columns=meanvar_rename)


def median_analysis(table: pd.core.groupby.GroupBy, variables: Set[str]) -> Iterator[pd.DataFrame]:
    """
    Returns two tables, one with medians and another with medians and other values
    """
    # create new column names
    medianvar_rename = {var: f"Median{var.capitalize()}" for var in variables}

    # table of medians
    yield table.median().rename(columns=medianvar_rename)
    # table of medians, stds, mins, maxes and counts
    yield table.aggregate(median_and_others).rename(columns=medianvar_rename)


def bonferroni_mark(pvalue: float, bonferroni_corr: float) -> str:
    """
    Prints the pvalue with 3 digits of precision and marks it's significance according to Bonferroni analysis
    """
    return format(pvalue, ".3f") + ("*" if pvalue < bonferroni_corr else "§" if pvalue < 0.05 else "")


def anova_analysis(table: pd.core.groupby.GroupBy, var: str) -> HolderTuple:
    """
    Returns the results for oneway ANOVA analysis in the table for the variable var
    """
    groups = (column for _, column in table[var])
    return anova_oneway(groups, use_var='equal', welch_correction=False)


def analyse(buf: TextIO, output_file: TextIO, variables: Set[str], analyses: List[List[str]]) -> None:
    """
    Performs statistical analyses on the table in buf and writes the results into output_file

    variables contains the column names that contains the variables to analyse

    analyses is a list of lists, each of which describe which column to group by
    """
    table = pd.read_table(buf, usecols=(
        ['specimenid', 'species', 'sex', 'locality'] + list(variables)))
    for analysis in analyses:
        do_analysis(table, variables, analysis, output_file)
        output_file.write("\n")


def do_analysis(table: pd.DataFrame, variables: Set[str], analysis: List[str], output_file: TextIO) -> None:
    """
    Performs statistical analyses on the table and writes the results into output_file

    variables contains the column names that contains the variables to analyse

    analysis is the list of columns to group by
    """
    # groupby doesn't behave as needed if analysis is empty
    groupedtable = table.groupby(analysis) if analysis else table

    for table in mean_analysis(groupedtable, variables):
        table.to_csv(output_file, float_format="%.3f", sep='\t')
        output_file.write("\n")

    for table in median_analysis(groupedtable, variables):
        table.to_csv(output_file, float_format="%.3f", sep='\t')
        output_file.write("\n")

    bonferroni_corr = 0.05 / len(variables)
    print('\t'.join(["Variable", "N valid cases", "Degrees of Freedom",
                     "F-value", "P (Significance)"]), file=output_file)
    for var in variables:
        anova = anova_analysis(groupedtable, var)
        print('\t'.join([var, str(anova.nobs_t), str(anova.df_num), format(
            anova.statistic, ".3f"), bonferroni_mark(anova.pvalue, bonferroni_corr)]), file=output_file)
