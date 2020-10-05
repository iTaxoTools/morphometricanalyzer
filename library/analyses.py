import itertools
import os
from contextlib import redirect_stdout
from typing import Dict, Iterator, List, Optional, Set, TextIO, Tuple

import numpy as np
import numpy.ma as ma
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kruskal, mannwhitneyu, ttest_ind
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from statsmodels.stats.base import HolderTuple
from statsmodels.stats.multicomp import MultiComparison, pairwise_tukeyhsd
from statsmodels.stats.oneway import anova_oneway


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


def analyse(buf: TextIO, output_file: TextIO, variables: List[str], analyses: List[List[str]]) -> None:
    """
    Performs statistical analyses on the table in buf and writes the results into output_file

    variables contains the column names that contains the variables to analyse

    analyses is a list of lists, each of which describe which column to group by
    """
    table = pd.read_table(buf, index_col='specimenid', usecols=(
        ['specimenid', 'species', 'sex', 'locality'] + variables))
    for analysis in analyses:
        do_analysis(table, variables, analysis, output_file)
        output_file.write("\n")


def bonferroni_note(count: int, corr: float) -> str:
    return f"Note: Applying a Bonferroni correction to the {count} separate ANOVA analyses(one for each of {count} measurements) reduced the significance level of 0.05 to {format_pvalue(corr)}. P values below 0.05 but larger than the Bonferroni corrected significance level are marked with §. P values that stay significant after applying the Bonferroni correction(values < Bonferroni-corrected significance level) are marked with an asterisk."


def blank_upper_triangle(table: np.array) -> np.array:
    return ma.MaskedArray(table, mask=~np.logical_not(np.triu(table)))


def do_analysis(table: pd.DataFrame, variables: List[str], analysis: List[str], output_file: TextIO) -> pd.DataFrame:
    """
    Performs statistical analyses on the table and writes the results into output_file

    variables contains the column names that contains the variables to analyse

    analysis is the list of columns to group by

    Returns a copy of the table with remarks
    """
    # without this something modifies the table
    # groupby doesn't behave as needed if analysis is empty
    groupedtable = table.groupby(analysis) if analysis else table
    bonferroni_corr = 0.05 / len(variables)

    print("1. Mean Analysis", file=output_file)
    for result in mean_analysis(groupedtable, variables):
        result.to_csv(output_file, float_format="%.3f",
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

    tukeyhsd_analysis(table, variables, analysis, output_file)

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
    for result in median_analysis(groupedtable, variables):
        result.to_csv(output_file, float_format="%.3f",
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

    print("6. Mann-Whitney U tests", file=output_file)
    print("U tests were implemented with continuity correction and two-tailed significances", file=output_file)
    full_table = []
    significance_table = []
    for ((group1_lbl, group1_table), (group2_lbl, group2_table)) in itertools.combinations(groupedtable, 2):
        row_label = (', '.join(group1_lbl) if isinstance(group1_lbl, tuple) else group1_lbl) + \
            ' - ' + (', '.join(group2_lbl)
                     if isinstance(group2_lbl, tuple) else group1_lbl)
        full_table.append(row_label)
        significance_table.append(row_label)
        for var in variables:
            u_val, pvalue = mannwhitneyu(
                group1_table[var], group2_table[var], alternative='two-sided')
            full_table[-1] += f"\tU = {u_val:.3f}, P = {bonferroni_mark(pvalue, bonferroni_corr)}"
            significance_table[-1] += f"\tP = {bonferroni_mark(pvalue, bonferroni_corr)}"
    print("\tMann-Whitney U tests, full test statistics", file=output_file)
    print("\t".join(["Variable"] + variables), file=output_file)
    for row in full_table:
        print(row, file=output_file)
    output_file.write('\n')
    print("\tMann-Whitney U tests, only significances (P)", file=output_file)
    print("\t".join(["Variable"] + variables), file=output_file)
    for row in significance_table:
        print(row, file=output_file)
    print(bonferroni_note(len(variables), bonferroni_corr), file=output_file)
    output_file.write('\n')

    print("7. Outliers", file=output_file)
    print("The following outlier values have been identified. These may simply indicate specimens with morphometric peculiarities, but could also be measurement or data transformation errors. Please check them carefully!", file=output_file)

    def is_outlier(col: pd.Series) -> pd.Series:
        q3 = col.quantile(0.75)
        q1 = col.quantile(0.25)
        iqr = q3 - q1
        return (col > (q3 + 1.5 * iqr)).combine(col < (q1 - 1.5 * iqr), lambda x, y: x or y)

    specimen_with_outliers: Dict[str, List[str]] = {}
    for var in variables:
        outlier_specimen = [specimenid for specimenid,
                            cond in groupedtable[var].transform(is_outlier).items() if cond]
        if outlier_specimen:
            print(f"{var}:", ', '.join(
                f"{specimenid} ({table[var][specimenid]})" for specimenid in outlier_specimen), file=output_file)

            for specimenid in outlier_specimen:
                specimen_with_outliers[specimenid] = specimen_with_outliers.setdefault(
                    specimenid, []) + [var]
    if specimen_with_outliers:
        for specimenid, outlier_vars in specimen_with_outliers.items():
            remark: str = str(table['remark'][specimenid])
            table['remark'][specimenid] = (remark + "; " if remark else "") + \
                f"Row contains outlier values ({', '.join(outlier_vars)})"

    table_with_species = table.rename(
        index=(lambda specimen: table['species'][specimen].replace(' ', '_')+'_'+specimen))
    output_file.write('\n')
    print("8. Euclidean distance", file=output_file)
    eucl_dist = pd.DataFrame(
        blank_upper_triangle(squareform(pdist(table_with_species[variables]))),
        index=table_with_species.index,
        columns=table_with_species.index
    )
    eucl_dist.to_csv(output_file, sep="\t",
                     float_format="%.2f", line_terminator="\n")

    output_file.write('\n')
    print("9. Cosine distance", file=output_file)
    eucl_dist = pd.DataFrame(
        blank_upper_triangle(squareform(
            pdist(table_with_species[variables], metric='cosine'))),
        index=table_with_species.index,
        columns=table_with_species.index
    )
    eucl_dist.to_csv(output_file, sep="\t",
                     float_format="%.2f", line_terminator="\n")
    output_file.write('\n')
    return table


def write_pca(table: pd.DataFrame, variables: List[str], output_file: TextIO) -> None:
    RETAINED_VARIANCE = 0.75
    MAX_PC = 6
    pca = PCA(RETAINED_VARIANCE)
    principal_components = pca.fit_transform(table[variables])
    pc_columns = [f"PC{i+1}" for i in range(0, pca.n_components_)]
    principalDf = pd.DataFrame(
        principal_components,
        index=table.index,
        columns=pc_columns
    )
    pd.concat([table['species'], principalDf], axis=1).to_csv(
        output_file, sep="\t", float_format="%.2f", line_terminator="\n",
        columns=['species'] + pc_columns[:MAX_PC])
    loading = pca.components_.T * np.sqrt(pca.explained_variance_)
    loading_matrix = pd.DataFrame(
        loading,
        index=variables,
        columns=pc_columns
    )
    loading_matrix.to_csv(
        output_file, sep="\t", float_format="%.3f", line_terminator="\n",
        columns=pc_columns[:MAX_PC])
    print('Explained variance',
          *[f"{ratio * 100:.1f}%" for ratio in pca.explained_variance_ratio_],
          sep='\t', file=output_file)
    output_file.write('\n')


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


def write_lda(table: pd.DataFrame, variables: List[str], analysis: List[str], output_file: TextIO) -> None:
    clf = LinearDiscriminantAnalysis()
    lda_table = clf.fit_transform(
        table[variables], table[analysis].agg(lambda l: '-'.join(l), axis=1))
    ld_num = lda_table.shape[1]
    ld_columns = [f"LD{i+1}" for i in range(0, ld_num)]
    principalDf = pd.DataFrame(
        lda_table,
        index=table.index,
        columns=ld_columns
    )
    prob_classes = clf.predict_proba(table[variables])
    prob_Df = pd.DataFrame(
        prob_classes,
        index=table.index,
        columns=[f"Prob {group}" for group in clf.classes_]
    )
    pd.concat([table[analysis], principalDf, prob_Df], axis=1).to_csv(
        output_file, sep="\t", float_format="%.2f", line_terminator="\n"
    )
    output_file.write('\n')


class Analyzer:
    """
    This class contains all the parameters for analysis
    """

    def __init__(self, buf: TextIO, variables: List[str], analyses: List[List[str]], output_file: TextIO, table_file: TextIO):
        self.table = pd.read_table(buf, index_col='specimenid', usecols=(
            ['specimenid', 'species', 'sex', 'locality'] + variables + ['remark']), dtype={'remark': 'string'})
        self.table['remark'].fillna("", inplace=True)
        self.output_file = output_file
        self.table_file = table_file
        self.analyses = analyses
        self.variables = variables
        self.size_var: Optional[str] = None

    def set_size_var(self, value: Optional[str] = None) -> None:
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
        size_var = self.size_var if self.size_var else self.variables[0]

        size_val = self.table[size_var]
        size_corr_renames = {
            var: f"ratio_{var}_{size_var}" for var in self.variables if var != size_var}
        size_corr_table = self.table.drop(columns=size_var).rename(
            columns=size_corr_renames)
        size_corr_variables = list(size_corr_renames.values())
        for var in size_corr_variables:
            size_corr_table[var] /= size_val
        for analysis in self.analyses:
            remarked_table = do_analysis(
                self.table.copy(), self.variables, analysis, self.output_file)
            self.output_file.write("\n")
            remarked_table.to_csv(
                self.table_file, sep='\t', line_terminator='\n')
            self.table_file.write('\n')

            self.output_file.write("Size corrected analysis\n")

            size_corr_table_remarked = do_analysis(
                size_corr_table.copy(), size_corr_variables, analysis, self.output_file)
            print("Linear discriminant analysis.", file=self.output_file)

            write_lda(size_corr_table, size_corr_variables,
                      analysis, self.output_file)
            size_corr_table_remarked.to_csv(
                self.table_file, sep='\t', line_terminator='\n')

        print("Principal component analysis.", file=self.output_file)

        write_pca(size_corr_table, size_corr_variables, self.output_file)

        print("Diagnoses.", file=self.output_file)

        size_corr_table.insert(loc=3, column=size_var,
                               value=self.table[size_var])
        order_species_ranges(
            size_corr_table, [size_var] + size_corr_variables, self.output_file)
