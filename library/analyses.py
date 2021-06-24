from contextlib import redirect_stdout
import itertools
import logging
import os, sys
import time
from typing import Any, Dict, Iterator, List, Optional, TextIO, Tuple
import warnings

import numpy as np
import numpy.ma as ma
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kruskal, mannwhitneyu, ttest_ind
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from statsmodels.stats.base import HolderTuple
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.oneway import anova_oneway

from library.process_plot import Plot

resource_path = getattr(sys, '_MEIPASS', sys.path[0])


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


def mean_analysis(table: Any, variables: List[str]) -> Iterator[pd.DataFrame]:
    """
    Returns two tables, one with means and another with means and other values
    """
    # create new column names
    meanvar_rename = {var: f"Mean{var.capitalize()}" for var in variables}

    # table of means
    yield table.mean().rename(columns=meanvar_rename)
    # table of means, stds, mins, maxes and counts
    yield table.aggregate(mean_and_others).rename(columns=meanvar_rename)


def median_analysis(table: Any, variables: List[str]) -> Iterator[pd.DataFrame]:
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


def anova_analysis(table: Any, var: str) -> HolderTuple:
    """
    Returns the results for oneway ANOVA analysis in the table for the variable var
    """
    groups = (column for _, column in table[var])
    return anova_oneway(groups, use_var='equal', welch_correction=False)


def tukeyhsd_analysis(table: pd.DataFrame, variables: List[str], analysis: List[str], output_file: TextIO) -> None:
    """
    Write the results of Tukey post-hos tests into the output_file

    Uses tukeyhsd and MultiComparison from statsmodel
    """
    groups = [",".join(group) for group in table[analysis].to_numpy()]
    group_order = sorted(set(groups))
    with open(os.devnull, mode="w") as devnull:
        # prevents tukeyhsd from writing garbage to the output
        with redirect_stdout(devnull):
            pvalues = [MultiComparison(table[var], groups, group_order=group_order).tukeyhsd(
            ).pvalues for var in variables]
    print("\tTukey Post-Hoc significance values", file=output_file)
    print("\t".join(["Variable"] + variables), file=output_file)
    for i, (group1, group2) in enumerate((group1, group2) for group1 in group_order for group2 in group_order if group1 < group2):
        print(f"{group1} - {group2}", *[format_pvalue(pvalue[i])
                                        for pvalue in pvalues], sep='\t', file=output_file)
    output_file.write("\n")


def bonferroni_note(count: int, corr: float) -> str:
    return f"Note: Applying a Bonferroni correction to the {count} separate ANOVA analyses(one for each of {count} measurements) reduced the significance level of 0.05 to {format_pvalue(corr)}. P values below 0.05 but larger than the Bonferroni corrected significance level are marked with §. P values that stay significant after applying the Bonferroni correction(values < Bonferroni-corrected significance level) are marked with an asterisk."


def blank_upper_triangle(table: np.array) -> np.array:
    return ma.MaskedArray(table, mask=~np.logical_not(np.triu(table)))




def order_species_ranges(table: pd.DataFrame, variables: List[str], output_file: TextIO) -> None:
    table = table.groupby('species').agg(lambda arr: (arr.min(), arr.max()))
    description: Dict[str, Dict[Tuple[str, str], List[str]]] = {
        species: {} for species in table.index}
    for species1 in table.index:
        for var in variables:
            for species2 in table.index:
                range1 = table[var][species1]
                range2 = table[var][species2]
                if range1[1] < range2[0]:
                    description[species1].setdefault(
                        ("smaller", var), []).append(species2)
                if range1[0] > range2[1]:
                    description[species1].setdefault(
                        ("larger", var), []).append(species2)

    def join_and(parts: List[str]) -> str:
        if len(parts) <= 1:
            return "".join(parts)
        else:
            return ", ".join(parts[:-1]) + " and " + parts[-1]

    def print_vs(species1: str, species2: str, var: str) -> str:
        range1 = table[var][species1]
        range2 = table[var][species2]
        return f"({range1[0]:.2f}-{range1[1]:.2f} vs. {range2[0]:.2f}-{range2[1]:.2f})"

    def list_species(species1: str, var: str, other: List[str]) -> str:
        return join_and([f"{species} {print_vs(species1, species, var)}" for species in other])

    for species, relations in description.items():
        print(species, file=output_file)
        output_file.write(f"{species} differs ")
        output_file.write(join_and([f"by a {relation} {var} from {list_species(species, var, other_species)}"
                                    for (relation, var), other_species in relations.items()]))
        output_file.write(".\n\n")




class Analyzer:
    """
    This class contains all the parameters for analysis
    """

    def __init__(self, buf: TextIO, variables: List[str], analyses: List[List[str]], table_file: TextIO, output_dir: str):
        """
        buf - buffer containing the data table
        variables - list of the names of fields that should be regarded as variables
        analyses - list of lists of name, each element of 'analyses' will be used for grouping in separate analyses
        output_file - buffer for the output of analyses' results
        table_file - buffer for the output of modified tables
        """
        self.table = pd.read_table(buf, index_col='specimenid', usecols=(
            ['specimenid', 'species', 'sex', 'locality'] + variables + ['remark']), dtype={'remark': 'string'},
            na_values={'specimenid': ""})
        if self.table.index.hasnans or not self.table.index.is_unique:
            raise ValueError("Problem detected with specimenid values: There are either duplicated or missing values. The analysis could not be executed. Please correct the error(s) in the input file.")
        self.table['remark'].fillna("", inplace=True)
        self.table_file = table_file
        self.analyses = analyses
        self.variables = variables
        self.size_var: Optional[str] = None
        self.start_time = time.monotonic()
        self.output_dir = output_dir
        self.plotter = Plot(output_dir)

    def output_file(self, normalized: bool, analysis: Optional[List[str]], name: str) -> TextIO:
        maybe_normalized = "_size_corrected" if normalized else ""
        if analysis:
            filename = f'{name}{maybe_normalized}_for_{"-".join(analysis)}.txt'
        else:
            filename = f'{name}{maybe_normalized}.txt'
        full_filename = os.path.join(self.output_dir, filename)
        return open(full_filename, mode="w")



    def log_with_time(self, message: str) -> None:
        time_passed = time.monotonic() - self.start_time
        logging.info(f"{time_passed:.1f}s: {message}")

    def set_size_var(self, value: Optional[str] = None) -> None:
        """
        Sets the name of the variable, which is used for normalization
        of the other variable during the second part of analysis

        Raises an ValueError if value is not one of the variables
        """
        if value:
            if value in self.variables:
                self.size_var = value
            else:
                raise ValueError(
                    f"Variable \"{value}\" is not present in the data")
        else:
            self.size_var = None

    def analyse(self) -> None:
        """
        Performs statistical analyses on self.table and writes the results into output_file
        """
        np.seterr('raise')  # enable detection of floating point errors

        # plot boxplots
        self.log_with_time("First boxplot")
        self.plotter.boxplot1(self.table)
        self.log_with_time("Second boxplot")
        self.plotter.boxplot2(self.table)
        self.log_with_time("Boxplots finished")

        # If normalization variable is not given, use the first variable
        size_var = self.size_var if self.size_var else self.variables[0]

        # column of values of the normalization variable
        size_val = self.table[size_var]

        # dictionary used to rename variable column in normalized tables
        size_corr_renames = {
            var: f"ratio_{var}_{size_var}" for var in self.variables if var != size_var}

        # the normalized table
        size_corr_table = self.table.drop(columns=size_var).rename(
            columns=size_corr_renames)

        # list of variables' names in the normalized table
        size_corr_variables = list(size_corr_renames.values())

        # perform the normalization
        for var in size_corr_variables:
            size_corr_table[var] /= size_val

        # perform the analyses that depend on grouping
        for current, analysis in enumerate(self.analyses):

            self.log_with_time(f"Analysis {current}")
            self.log_with_time("Uncorrected analysis")
            # copy of self.table with remarks specific to the current grouping
            remarked_table = self.do_analysis(
                self.table.copy(), self.variables, analysis, normalized=False)  # results of non-normalized analyses are written to the output file

            # write the remarked table to the table file
            remarked_table.to_csv(
                self.table_file, sep='\t', line_terminator='\n')
            self.table_file.write('\n')

            # results of normalized analyses are written to the output file
            self.log_with_time("Size corrected analysis")
            # copy of size_corr_table with remarks
            size_corr_table_remarked = self.do_analysis(
                size_corr_table.copy(), size_corr_variables, analysis, normalized=True)

            # write the remarked normalized table to the table file
            size_corr_table_remarked.to_csv(
                self.table_file, sep='\t', line_terminator='\n')

        # reinsert size_var back into the normalized table for the remaining analyses
        size_corr_table.insert(loc=3, column=size_var,
                               value=self.table[size_var])

        for current, analysis in enumerate(self.analyses):
            with self.output_file(normalized=True, analysis=analysis, name="LDA") as output_file:
                print("Linear discriminant analysis.\n", file=output_file)
                print("This file shows the results of a Linear Discriminant Analysis carried out on the size-corrected data plus the size variable. The data shown are the values of the LDA components for each sample.\n", file=output_file)
                self.log_with_time(f"Linear Discriminant analysis {current}")
                # The result of Linear Discriminant analysis on normalized variables is written to the output file
                self.write_lda(size_corr_table, [size_var] + size_corr_variables,
                          analysis, output_file)

        # The result of Principal Component analysis on normalized variables is written to the output file
        with self.output_file(normalized=False, analysis=None, name="PCA") as output_file:
            self.log_with_time("Principal component analysis")
            print("Principal component analysis.\n", file=output_file)
            print("This file shows the results of a Principal Component Analysis carried out on the size-corrected data plus the size variable. The data shown are the values of the first four Principal Components (PCs) for each sample, and the factor loadings and percent explained variance for each variable in the analysis. All variables were Min-Max normalized before analysis.\n", file=output_file)
            self.write_pca(size_corr_table, [size_var] + size_corr_variables, output_file)

        # PCA without scaling
        with self.output_file(normalized=False, analysis=None, name="PCA_no_MinMax_normalization") as output_file:
            self.log_with_time("Principal component analysis")
            print("Principal component analysis.\n", file=output_file)
            self.write_pca(size_corr_table, [size_var] + size_corr_variables, output_file, scale=False, graph=False)

        self.log_with_time("Diagnoses")
        with self.output_file(normalized=False, analysis=None, name="Diagnoses") as output_file:
            print("Diagnoses.\n", file=output_file)
            print("This file lists all cases of non-overlapping ranges of values for the size-corrected variables and the size variable in text format.\n", file=output_file)

            # searches for instances of pairs of species with non-overlapping ranges in some variable and displays them in the output file
            order_species_ranges(
                size_corr_table, [size_var] + size_corr_variables, output_file)
        self.log_with_time("Analysis completed")


    def do_analysis(self, table: pd.DataFrame, variables: List[str], analysis: List[str], normalized: bool) -> pd.DataFrame:
        """
        Performs statistical analyses on the table and writes the results into output_file

        variables contains the column names that contains the variables to analyse

        analysis is the list of columns to group by

        Returns a copy of the table with remarks
        """
        # groupby doesn't behave as needed if analysis is empty
        groupedtable = table.groupby(analysis) if analysis else table

        # make table with groups >= 2 for some analyses
        small_groups = groupedtable.transform(lambda group: group.count() >= 2).iloc[:, 0]
        groupedtable_filtered = table.loc[small_groups].groupby(analysis) if analysis else table.loc[small_groups]
        if len(groupedtable_filtered.groups) < 2:
            groupedtable_filtered = None

        bonferroni_corr = 0.05 / len(variables)

        self.log_with_time("1. Mean Analysis")
        with self.output_file(normalized, analysis, "Mean_analysis") as output_file:
            print("1. Mean Analysis\n", file=output_file)
            for result in mean_analysis(groupedtable, variables):
                result.to_csv(output_file, float_format="%.3f",
                        sep='\t', line_terminator='\n')
                output_file.write("\n")

        # Here anova_analysis is used as a wrapper for anova_oneway from statsmodel.
        # For each variable, ANOVA analysis is performed on the corresponding column,
        # yielding a tuple of results (HolderTuple class from statsmodel).
        # Floating point errors in ANOVA are caught and abort the analysis.
        # The tuple is then used to construct of the output table.
        # After the loop the note about Bonferroni correction is printed

        with self.output_file(normalized, analysis, "Simple_ANOVA") as output_file:
            # Header for the ANOVA analysis
            self.log_with_time("2. Simple ANOVA")
            print("2. Simple ANOVA\n", file=output_file)
            if groupedtable_filtered:
                print('\t'.join(["Variable", "N valid cases", "Degrees of Freedom",
                "F-value", "P (Significance)"]), file=output_file)

                for var in variables:
                    try:
                        # contains the result of anova_oneway
                        anova = anova_analysis(groupedtable_filtered, var)
                    except FloatingPointError:
                        print("Error: Invalid data for simple ANOVA", file=output_file)
                        warnings.warn("Floating point error in simple ANOVA",
                                RuntimeWarning)
                        break
                    # print a line for var with ANOVA results
                    print('\t'.join([var, str(anova.nobs_t), str(anova.df_num), format(
                        anova.statistic, ".3f"), bonferroni_mark(anova.pvalue, bonferroni_corr)]), file=output_file)
                    # print the note about bonferroni correction
                print(bonferroni_note(len(variables), bonferroni_corr), file=output_file)
            else:
                print("Not enough data for simple ANOVA", file=output_file)
            output_file.write("\n")

            self.log_with_time("Tukey post-hoc analysis")
            # tukeyhsd_analysis perform the Tukey analysis and write the result into the output file
            tukeyhsd_analysis(table, variables, analysis, output_file)

        # Student's t-test is performed on pairs of subtables for grouping label
        # For each pair the t-test return two Iterable for statistic values and pvalues
        # The floating point error are caught and abort the analysis
        # The a row of the table is printed
        # The label of the row is constructed from the current pair of subtable labels
        # The statistic values and pvalues are paired up and written to the cells of the row
        # The note about Bonferroni correction is printed after the table
        # uses ttest_ind from scipy.stats

        with self.output_file(normalized, analysis, "Students_t_test") as output_file:
            # Header for the Student's t-test analysis
            self.log_with_time("3. Student's t-test")
            print("3. Student's t-test\n", file=output_file)
            if groupedtable_filtered:
                print("\tStudent's t-test", file=output_file)
                print('\t'.join(['Variable'] + variables), file=output_file)

                # Iteration of all pairs of groups
                for ((group1_lbl, group1_table), (group2_lbl, group2_table)) in itertools.combinations(groupedtable_filtered, 2):
                    try:
                        # makes two lists of statistics and p-values with entries for each variable
                        statistics, pvalues = ttest_ind(group1_table[sorted(
                            variables)], group2_table[variables], nan_policy='omit')
                    except FloatingPointError:
                        print("Error: Invalid data for the Student's t-test", file=output_file)
                        warnings.warn(
                                "Floating point error in the Student's t-test", category=RuntimeWarning)
                        break
                    # compose label for the current Student's test
                    # for example:
                    # species1 - species2
                    # or:
                    # species1, locality1 - species2, locality2
                    row_label = (', '.join(group1_lbl) if isinstance(group1_lbl, tuple) else group1_lbl) + \
                            ' - ' + (', '.join(group2_lbl)
                                    if isinstance(group2_lbl, tuple) else group2_lbl)
                            # makes a row of
                    # statistic(var1), pvalue(var1)<Tab>...
                    row_content = '\t'.join(
                            f"t = {statistic:.3f}; P = {bonferroni_mark(pvalue, bonferroni_corr)}" for statistic, pvalue in zip(statistics, pvalues))
                    print(row_label, row_content, sep='\t', file=output_file)

                # Prints a note about the bonferroni correction
                print(bonferroni_note(len(variables), bonferroni_corr), file=output_file)
            else:
                print("Not enough data for Student's t-test", file=output_file)
            output_file.write('\n')

        with self.output_file(normalized, analysis, "Median_analysis") as output_file:
            self.log_with_time("4. Median Analysis")
            print("4. Median Analysis\n", file=output_file)
            for result in median_analysis(groupedtable, variables):
                result.to_csv(output_file, float_format="%.3f",
                        sep='\t', line_terminator='\n')
                output_file.write("\n")

        # For each variable Kruskal-Wallis analysis is performed on the groups of values in the corresponding column
        # It return a statistic value and a pvalue,
        # which are then printed with the variable name as a table row
        # After the loop the note about Bonferroni correction is written
        with self.output_file(normalized, analysis, "Kruskal_Wallis_ANOVA") as output_file:
            self.log_with_time("5. Kruskal-Wallis ANOVA")
            print("5. Kruskal-Wallis ANOVA\n", file=output_file)
            if groupedtable_filtered:
                print("Variable", "N valid cases",
                        "P (significance)", sep='\t', file=output_file)
                for var in variables:
                    statistic, pvalue = kruskal(
                            *(values for _, values in groupedtable_filtered[var]), nan_policy='omit')
                    print(var, f"H() = {statistic:.3f}",
                            f"p = {bonferroni_mark(pvalue, bonferroni_corr)}", sep='\t', file=output_file)
                    print(bonferroni_note(len(variables), bonferroni_corr), file=output_file)
            else:
                print("Not enough data for Kruskal-Wallis ANOVA", file=output_file)
            output_file.write('\n')

        # Two output tables need to be written but since the output file can only be written sequentially,
        # the lines of the tables are temporarily stored in two lists of strings
        # The output tables are generated through iteration over pairs of subtables
        # The label for the corresponding rows is generated from subtable labels and appended to both lists
        # Then the results of Mann-Whitney test is appended to the last element of the lists in two different forms
        # After the loop the lines from the lists are printed sequentially
        # Finally, the note about the Bonferroni correction is printed
        # uses mannwhitneyu from scipy.stats

        with self.output_file(normalized, analysis, "Mann_Whitney_u_test") as output_file:
            self.log_with_time("6. Mann-Whitney U tests")
            print("6. Mann-Whitney U tests\n", file=output_file)
            if groupedtable_filtered:
                print("U tests were implemented with continuity correction and two-tailed significances", file=output_file)
                # the lines of the first table
                full_table = []
                # the lines of the second table
                significance_table = []
                for ((group1_lbl, group1_table), (group2_lbl, group2_table)) in itertools.combinations(groupedtable_filtered, 2):
                    row_label = (', '.join(group1_lbl) if isinstance(group1_lbl, tuple) else group1_lbl) + \
                            ' - ' + (', '.join(group2_lbl)
                                    if isinstance(group2_lbl, tuple) else group2_lbl)
                            # Each table gets the same row label
                    full_table.append(row_label)
                    significance_table.append(row_label)
                    for var in variables:
                        u_val, pvalue = mannwhitneyu(
                                group1_table[var], group2_table[var], alternative='two-sided')
                        # the first table gets both u_value and pvalue
                        full_table[-1] += f"\tU = {u_val:.3f}, P = {bonferroni_mark(pvalue, bonferroni_corr)}"
                        # the second table
                        significance_table[-1] += f"\tP = {bonferroni_mark(pvalue, bonferroni_corr)}"

                # print the first table
                print("\tMann-Whitney U tests, full test statistics", file=output_file)
                print("\t".join(["Variable"] + variables), file=output_file)
                for row in full_table:
                    print(row, file=output_file)
                output_file.write('\n')
                # print the second table
                print("\tMann-Whitney U tests, only significances (P)", file=output_file)
                print("\t".join(["Variable"] + variables), file=output_file)
                for row in significance_table:
                    print(row, file=output_file)
                # print the note about the Bonferroni correction
                print(bonferroni_note(len(variables), bonferroni_corr), file=output_file)
            else:
                print("Not enough data for Mann-Whitney U tests", file=output_file)
            output_file.write('\n')

        # For each variable, the specimens which are outlier with respect to this variable are first printed as a table row
        # then they are collected into the dictionary mapping each specimen id to the variable in which it's a outlier
        # After the loop over the variable this dictionary is used to construct remarks in the data table
        with self.output_file(normalized, analysis, "Outliers") as output_file:
            self.log_with_time("7. Outliers")
            print("7. Outliers\n", file=output_file)
            print("The following outlier values have been identified. These may simply indicate specimens with morphometric peculiarities, but could also be measurement or data transformation errors. Please check them carefully!", file=output_file)

            def is_outlier(col: pd.Series) -> pd.Series:
                """
                Takes a column of values and return a column of Booleans with True marking the outliers
                """
                q3 = col.quantile(0.75)
                q1 = col.quantile(0.25)
                iqr = q3 - q1
                return (col > (q3 + 1.5 * iqr)) | (col < (q1 - 1.5 * iqr))

            specimen_with_outliers: Dict[str, List[str]] = {}
            for var in variables:
                outlier_specimen = [specimenid for specimenid,
                        cond in groupedtable[var].transform(is_outlier).items() if cond]
                if outlier_specimen:
                    # print the row of the outlier table
                    print(f"{var}:", ', '.join(
                        f"{specimenid} ({table.at[specimenid, var]})" for specimenid in outlier_specimen), file=output_file)

                    # add the variable to the outlier dictionary
                    for specimenid in outlier_specimen:
                        specimen_with_outliers[specimenid] = specimen_with_outliers.setdefault(
                                specimenid, []) + [var]
                        # write remarks about outliers to the data table
            if specimen_with_outliers:
                for specimenid, outlier_vars in specimen_with_outliers.items():
                    remark: str = str(table['remark'][specimenid])
                    table['remark'][specimenid] = (remark + "; " if remark else "") + \
                            f"Row contains outlier values ({', '.join(outlier_vars)})"
            output_file.write('\n')

        # relabels the table index from specimenid to specimenid_species
        table_with_species = table.rename(
                index=(lambda specimen: table['species'][specimen].replace(' ', '_')+'_'+str(specimen)))

        # Next two parts construct tables with distances
        # Each use squareform and pdist from scipy.spatial to construct the table of distance
        # Then blank_upper_triangle function removes the values above the diagonal
        # Then the index and column labels are added from the data table
        # Finally, the distance table is printed
        with self.output_file(normalized, analysis, "Distances") as output_file:
            self.log_with_time("8. Euclidean distance")
            if normalized:
                print("This file displays matrices of Euclidean and Cosine distances among individuals, calculated from the size-corrected morphometric data.\n", file=output_file)
            else:
                print("This file displays matrices of Euclidean and Cosine distances among individuals, calculated from the raw (not size-corrected) morphometric data.\n", file=output_file)
            print("8. Euclidean distance\n", file=output_file)
            eucl_dist = pd.DataFrame(
                    blank_upper_triangle(squareform(pdist(table_with_species[variables]))),
                    index=table_with_species.index,
                    columns=table_with_species.index
                    )
            eucl_dist.to_csv(output_file, sep="\t",
                    float_format="%.2f", line_terminator="\n")
            output_file.write('\n')

            self.log_with_time("9. Cosine distance")
            print("9. Cosine distance\n", file=output_file)
            eucl_dist = pd.DataFrame(
                    blank_upper_triangle(squareform(
                        pdist(table_with_species[variables], metric='cosine'))),
                    index=table_with_species.index,
                    columns=table_with_species.index
                    )
            eucl_dist.to_csv(output_file, sep="\t",
                    float_format="%.2f", line_terminator="\n")
            output_file.write('\n')

        # return the data table with remarks
        return table


    def write_lda(self, table: pd.DataFrame, variables: List[str], analysis: List[str], output_file: TextIO) -> None:
        clf = LinearDiscriminantAnalysis()
        try:
            lda_table = clf.fit_transform(
                table[variables], table[analysis].agg(lambda l: '-'.join(l), axis=1))
        except FloatingPointError:
            print("Error: Invalid data for Linear Discriminant Analysis", file=output_file)
            warnings.warn("Floating point error in Linear Discriminant Analysis")
            return
        ld_num = lda_table.shape[1]
        ld_columns = [f"LD{i+1}" for i in range(0, ld_num)]
        principalDf = pd.DataFrame(
            lda_table,
            index=table.index,
            columns=ld_columns
        )
        labels = pd.Series(table[analysis[0]], index = table.index, name="-".join(analysis)).str.cat(table[analysis[1:]], sep='-')
        self.log_with_time("Starting LD plot")
        self.plotter.ldaplot(pd.concat([labels, principalDf], axis=1))
        self.log_with_time("Finished LD plot")
        # try:
        #     prob_classes = clf.predict_proba(table[variables])
        # except FloatingPointError:
        #     warnings.warn(
        #         "Floating point error in Linear Discriminant Analysis probability prediction", category=RuntimeWarning)
        #     pd.concat([table[analysis], principalDf], axis=1).to_csv(
        #         output_file, sep="\t", float_format="%.2f", line_terminator="\n"
        #     )
        # else:
        #     prob_Df = pd.DataFrame(
        #         prob_classes,
        #         index=table.index,
        #         columns=[f"Prob {group}" for group in clf.classes_]
        #     )
        #     pd.concat([table[analysis], principalDf, prob_Df], axis=1).to_csv(
        #         output_file, sep="\t", float_format="%.2f", line_terminator="\n"
        #     )
        pd.concat([table[analysis], principalDf], axis=1).to_csv(
            output_file, sep="\t", float_format="%.2f", line_terminator="\n"
        )
        output_file.write('\n')


    def write_pca(self, table: pd.DataFrame, variables: List[str], output_file: TextIO, scale:bool=True, graph:bool=True) -> None:
        RETAINED_VARIANCE = 0.75
        MIN_PC = 4
        MAX_PC = 6
        pca = PCA()
        if scale:
            scaled_table = MinMaxScaler().fit_transform(table[variables])
        else:
            scaled_table = table[variables]
        principal_components = pca.fit_transform(scaled_table)
        explained_variance = 0
        for i, variance in enumerate(pca.explained_variance_ratio_):
            explained_variance += variance
            if explained_variance >= RETAINED_VARIANCE:
                pca_components = max(i+1, MIN_PC)
                break
        else:
            pca_components = pca.n_components_
        pc_columns = [f"PC{i+1}" for i in range(0, pca_components)]
        principalDf = pd.DataFrame(
            principal_components[:, :pca_components],
            index=table.index,
            columns=pc_columns
        )
        pca_table = pd.concat([table['species'], principalDf], axis=1)
        pca_table.to_csv(
            output_file, sep="\t", float_format="%.2f", line_terminator="\n",
            columns=['species'] + pc_columns[:MAX_PC])
        if graph:
            self.log_with_time("Starting PCA plot")
            self.plotter.pcaplot(pca_table)
            self.log_with_time("Finished PCA plot")
        loading = pca.components_.T * np.sqrt(pca.explained_variance_)
        loading_matrix = pd.DataFrame(
            loading[:, :pca_components],
            index=variables,
            columns=pc_columns
        )
        loading_matrix.to_csv(
            output_file, sep="\t", float_format="%.3f", line_terminator="\n",
            columns=pc_columns[:pca_components])
        print('Explained variance',
              *[f"{ratio * 100:.1f}%" for ratio in pca.explained_variance_ratio_],
              sep='\t', file=output_file)
        output_file.write('\n')
