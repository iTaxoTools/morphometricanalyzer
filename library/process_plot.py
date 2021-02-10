import os

import matplotlib
from matplotlib import rc
from matplotlib.patches import PathPatch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings



def adjust_box_widths(boxplot: sns.FacetGrid, fac: float) -> None:
    """
    Adjust the withs of a seaborn-generated boxplot.
    https://stackoverflow.com/questions/31498850/set-space-between-boxplots-in-python-graphs-generated-nested-box-plots-with-seab
    """

    # iterating through Axes instances
    for ax in boxplot.axes.flatten():

        # iterating through axes artists:
        for c in ax.get_children():

            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5*(xmin+xmax)
                xhalf = 0.5*(xmax - xmin)

                # setting new width of box
                xmin_new = xmid-fac*xhalf
                xmax_new = xmid+fac*xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])


class Plot:
    """
    This class contains all the parameters for plots
    """

    def __init__(self, output_dir: str):
        """
        buf - buffer containing the data table
        y- path string of directory for plots output

        """
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42

        rc('font', **{'family': 'sans-serif'})
        self.output_dir = output_dir

    def boxplot1(self, df: pd.DataFrame):
        xx = sorted(df['species'].unique(), reverse=True)
        numeric_columns = (col for col in df.columns if pd.api.types.is_numeric_dtype(df.dtypes[col]))
        for col in numeric_columns:
            g = sns.FacetGrid(df[["species", col]], height=8, aspect=1)
            g.map(sns.boxplot, "species", col,
                  orient="v", palette="Set2", order=xx)
            g.map(sns.stripplot, "species", col,
                  jitter=True, color="grey", order=xx)
            #adjust_box_widths(g, 0.9)
            for ax in g.axes.flat:
                ax.set_xlabel(ax.get_xlabel(), fontsize=12)
                #ax.set_title(ax.get_title(),  fontsize=20)
                ax.set_ylabel(ax.get_ylabel(), fontsize=12)
                ax.tick_params(axis="x", labelsize=10)
                ax.tick_params(axis="y", labelsize=10)

            plt.savefig(os.path.join(self.output_dir, col +
                                     "_species_boxplot"+'.pdf'), transparent=True)

    def boxplot2(self, df: pd.DataFrame):
        xx = sorted(df['sex'].unique(), reverse=True)
        numeric_columns = (col for col in df.columns if pd.api.types.is_numeric_dtype(df.dtypes[col]))
        for col in numeric_columns:
            g = sns.FacetGrid(df[['species', 'sex', col]], col="species", sharex=False,
                              sharey=False, height=4)
            g.map(sns.boxplot, "sex", col, orient="v", palette="Set2", order=xx)
            g.map(sns.stripplot, "sex", col, jitter=True, color=".25", order=xx)
            adjust_box_widths(g, 0.9)
            for ax in g.axes.flat:
                ax.set_xlabel(ax.get_xlabel(), fontsize=12)
                ax.set_title(ax.get_title(),  fontsize=12)
                ax.set_ylabel(ax.get_ylabel(), fontsize=12)
            plt.savefig(os.path.join(self.output_dir, col +
                                     "_sex_paired_boxplot"+'.pdf'), transparent=True)

    def pcaplot(self, df: pd.DataFrame):
        """
        Takes a dataframe of principal components analysis results
        with columns:
        species, PC1, PC2, ...
        """
        keys = df['species'].unique()
        targets = keys
        colors= sns.color_palette("Dark2", 9)
        markers=["*", "o", "s", "D"]
        n_components = min(len(df.columns) - 1, 4);
        if n_components < 2:
            warnings.warn("Less than 2 PCA components have been calculated.\nPCA plot is not possible")
            return

        for c1, c2 in ((1, 2), (2, 3), (3, 4)):
            fig = plt.figure(figsize = (8,8))
            ax = fig.add_subplot(1,1,1)
            ax.set_xlabel(f'Principal Component {c1}', fontsize = 15)
            ax.set_ylabel(f'Principal Component {c2}', fontsize = 15)
            ax.set_title('2 component PCA', fontsize = 20)

            for target, color, marker in zip(targets, colors, markers):
                indicesToKeep = df['species'] == target
                ax.scatter(df.loc[indicesToKeep, f'PC{c1}']
                           , df.loc[indicesToKeep, f'PC{c2}'], color = color, marker= marker
                           , s = 50, label=target)
            ax.tick_params(axis="x", labelsize=15)
            ax.tick_params(axis="y", labelsize=15)
            ax.set_xlabel(ax.get_xlabel(),   fontsize=18)
            ax.set_title(ax.get_title(), fontsize=20)
            ax.set_ylabel(ax.get_ylabel(), fontsize=18)
            ax.legend(loc= "best", prop=dict(size=15))
            plt.savefig(os.path.join(self.output_dir, f"pca{c1}{c2}_plot"+'.pdf'), transparent=True)

    def ldaplot(self, df: pd.DataFrame):
        """
        Takes a LDA table with columns
        [groups] LD1 LD2 ...
        """
        label = df.columns[0]
        targets = df[label].unique()
        colors= sns.color_palette("Dark2", 9)
        markers=["*", "o", "s", "D"]

        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(1,1,1)
        if len(df.columns) < 3:
            warnings.warn(f"LDA plot for {label} is not possible")
            return

        for target, color, marker in zip(targets, colors, markers):
            indicesToKeep = df[label] == target
            ax.scatter(df.loc[indicesToKeep, 'LD1']
                       , df.loc[indicesToKeep, 'LD2']
                       , color = color, marker= marker
                       , s = 50, label=target)
        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=15)
        ax.set_xlabel(ax.get_xlabel(),   fontsize=18)
        ax.set_title(ax.get_title(), fontsize=20)
        ax.set_ylabel(ax.get_ylabel(), fontsize=18)
        ax.set_xlabel('LDA 1', fontsize = 15)
        ax.set_ylabel('LDA 2', fontsize = 15)
        ax.set_title('2 Component LDA Plot', fontsize = 20)
        ax.legend(loc= "best", prop=dict(size=15))

        plt.savefig(os.path.join(self.output_dir, f"{label}_lda_plot"+'.pdf'), transparent=True)
