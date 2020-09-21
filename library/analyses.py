from typing import Iterator, List, Set, TextIO

import pandas as pd
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
from statsmodels.stats.base import HolderTuple
from scipy.stats import ttest_ind, kruskal
import os
from contextlib import redirect_stdout
import itertools


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


def mean_analysis(table: pd.core.groupby.GroupBy, variables: List[str]) -> Iterator[pd.DataFrame]:
    """
    Returns two tables, one with means and another with means and other values
    """
    # create new column names
    meanvar_rename = {var: f"Mean{var.capitalize()}" for var in variables}

    # table of means
    yield table.mean().rename(columns=meanvar_rename)
    # table of means, stds, mins, maxes and counts
    yield table.aggregate(mean_and_others).rename(columns=meanvar_rename)


def median_analysis(table: pd.core.groupby.GroupBy, variables: List[str]) -> Iterator[pd.DataFrame]:
    """
    Returns two tables, one with medians and another with medians and other values
    """
    # create new column names
    medianvar_rename = {var: f"Median{var.capitalize()}" for var in variables}

    # table of medians
    yield table.median().rename(columns=medianvar_rename)
    # table of medians, stds, mins, maxes and counts
    yield table.aggregate(median_and_others).rename(columns=medianvar_rename)


def format_pvalue(pvalue: float) -> str:
    """
    Formats a pvalue up to 5 or 10 decimals depending on its size, or shows that it's too small.
    """
    if pvalue < 1e-10:
        return f"<{1e-10:.10f}"
    elif pvalue < 1e-5:
        return format(pvalue, ".10f")
    else:
        return format(pvalue, ".5f")


def bonferroni_mark(pvalue: float, bonferroni_corr: float) -> str:
    """
    Prints the pvalue with 3 digits of precision and marks it's significance according to Bonferroni analysis
    """
    return format_pvalue(pvalue) + ("*" if pvalue < bonferroni_corr else "§" if pvalue < 0.05 else "")


def anova_analysis(table: pd.core.groupby.GroupBy, var: str) -> HolderTuple:
    """
    Returns the results for oneway ANOVA analysis in the table for the variable var
    """
    groups = (column for _, column in table[var])
    return anova_oneway(groups, use_var='equal', welch_correction=False)


def tukeyhsd_analysis(table: pd.DataFrame, variables: List[str], analysis: List[str], output_file: TextIO) -> None:
    """
    Write the results of Tukey post-hos tests into the output_file
    """
    groups = [",".join(group) for group in table[analysis].to_numpy()]
    group_order = sorted(set(groups))
    with open(os.devnull, mode="w") as devnull:
        with redirect_stdout(devnull):
            pvalues = [MultiComparison(table[var], groups, group_order=group_order).tukeyhsd(
            ).pvalues for var in variables]
    print("\tTukey Post-Hoc significance values", file=output_file)
    print("\t".join(["Variable"] + variables), file=output_file)
    for i, (group1, group2) in enumerate((group1, group2) for group1 in group_order for group2 in group_order if group1 < group2):
        print(f"{group1} - {group2}", *[format_pvalue(pvalue[i])
                                        for pvalue in pvalues], sep='\t', file=output_file)
    output_file.write("\n")


def analyse(buf: TextIO, output_file: TextIO, variables: Set[str], analyses: List[List[str]]) -> None:
    """
    Performs statistical analyses on the table in buf and writes the results into output_file

    variables contains the column names that contains the variables to analyse

    analyses is a list of lists, each of which describe which column to group by
    """
    table = pd.read_table(buf, usecols=(
        ['specimenid', 'species', 'sex', 'locality'] + list(variables)))
    for analysis in analyses:
        do_analysis(table, sorted(variables), analysis, output_file)
        output_file.write("\n")


def bonferroni_note(count: int, corr: float) -> str:
    return f"Note: Applying a Bonferroni correction to the {count} separate ANOVA analyses(one for each of {count} measurements) reduced the significance level of 0.05 to {corr}. P values below 0.05 but larger than the Bonferroni corrected significance level are marked with §. P values that stay significant after applying the Bonferroni correction(values < Bonferroni-corrected significance level) are marked with an asterisk."


def do_analysis(table: pd.DataFrame, variables: List[str], analysis: List[str], output_file: TextIO) -> None:
    """
    Performs statistical analyses on the table and writes the results into output_file

    variables contains the column names that contains the variables to analyse

    analysis is the list of columns to group by
    """
    # without this something modifies the table
    tukeytable = table.copy()
    # groupby doesn't behave as needed if analysis is empty
    groupedtable = table.groupby(analysis) if analysis else table
    bonferroni_corr = 0.05 / len(variables)

    print("1. Mean Analysis", file=output_file)
    for table in mean_analysis(groupedtable, variables):
        table.to_csv(output_file, float_format="%.3f",
                     sep='\t', line_terminator='\n')
        output_file.write("\n")

    print("2. Simple ANOVA", file=output_file)
    print('\t'.join(["Variable", "N valid cases", "Degrees of Freedom",
                     "F-value", "P (Significance)"]), file=output_file)
    for var in variables:
        anova = anova_analysis(groupedtable, var)
        print('\t'.join([var, str(anova.nobs_t), str(anova.df_num), format(
            anova.statistic, ".3f"), bonferroni_mark(anova.pvalue, bonferroni_corr)]), file=output_file)
    print(bonferroni_note(len(variables), bonferroni_corr), file=output_file)
    output_file.write("\n")

    tukeyhsd_analysis(tukeytable, variables, analysis, output_file)

    print("3. Student's t-test", file=output_file)
    print("\tStudent's t-test", file=output_file)
    print('\t'.join(['Variable'] + variables), file=output_file)
    for ((group1_lbl, group1_table), (group2_lbl, group2_table)) in itertools.combinations(groupedtable, 2):
        statistics, pvalues = ttest_ind(group1_table[sorted(
            variables)], group2_table[variables], nan_policy='omit')
        row_label = (', '.join(group1_lbl) if isinstance(group1_lbl, tuple) else group1_lbl) + \
            ' - ' + (', '.join(group2_lbl)
                     if isinstance(group2_lbl, tuple) else group1_lbl)
        row_content = '\t'.join(
            f"t = {statistic:.3f}; P = {bonferroni_mark(pvalue, bonferroni_corr)}" for statistic, pvalue in zip(statistics, pvalues))
        print(row_label, row_content, sep='\t', file=output_file)
    print(bonferroni_note(len(variables), bonferroni_corr), file=output_file)
    output_file.write('\n')

    print("4. Median Analysis", file=output_file)
    for table in median_analysis(groupedtable, variables):
        table.to_csv(output_file, float_format="%.3f",
                     sep='\t', line_terminator='\n')
        output_file.write("\n")

    print("5. Kruskal-Wallis ANOVA", file=output_file)
    print("Variable", "N valid cases",
          "P (significance)", sep='\t', file=output_file)
    for var in variables:
        statistic, pvalue = kruskal(
            *(values for _, values in groupedtable[var]), nan_policy='omit')
        print(var, f"H() = {statistic:.3f}",
              f"p = {bonferroni_mark(pvalue, bonferroni_corr)}", sep='\t', file=output_file)
    print(bonferroni_note(len(variables), bonferroni_corr), file=output_file)
    output_file.write('\n')
