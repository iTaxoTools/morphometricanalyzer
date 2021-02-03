import sys
import io
import warnings
from library.mistake_corrector import *
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif'})

from matplotlib.patches import PathPatch

def adjust_box_widths(g, fac):
    """
    Adjust the withs of a seaborn-generated boxplot.
    https://stackoverflow.com/questions/31498850/set-space-between-boxplots-in-python-graphs-generated-nested-box-plots-with-seab
    """

    ##iterating through Axes instances
    for ax in g.axes.flatten():

        ##iterating through axes artists:
        for c in ax.get_children():

            ##searching for PathPatches
            if isinstance(c, PathPatch):
                ##getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:,0])
                xmax = np.max(verts_sub[:,0])
                xmid = 0.5*(xmin+xmax)
                xhalf = 0.5*(xmax - xmin)

                ##setting new width of box
                xmin_new = xmid-fac*xhalf
                xmax_new = xmid+fac*xhalf
                verts_sub[verts_sub[:,0] == xmin,0] = xmin_new
                verts_sub[verts_sub[:,0] == xmax,0] = xmax_new

                ##setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin,xmax]):
                        l.set_xdata([xmin_new,xmax_new])


def processplot(x):
    """
    Function taking input file (tab, or any typical file allowed by morphometricanalyzer tool) using MistakeCorrector class by Vladimir
    """
    with open(x) as input_file:
        corrector = MistakeCorrector(input_file)
        buf = io.StringIO()
        for line in corrector:
            print(line, file=buf)
        buf.seek(0, 0)
        Plot(buf)




class Plot:
    """
    This class contains all the parameters for plots
    """

    def __init__(self, buf: TextIO):
        """
        buf - buffer containing the data table
        
        """
        self.table = pd.read_table(buf, index_col='specimenid')
        self.table['remark'].fillna("", inplace=True)

        self.pcaplot()
        self.plsplot()
        self.boxplot2()
        self.boxplot1()



    def boxplot1(self):
        df= self.table
        df.drop(["remark"], axis= 1)
        xx= sorted(df['species'].unique(), reverse= True)
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        df1 = df.select_dtypes(include=numerics)
        for i, col in enumerate(df1.columns):

            g= sns.FacetGrid(df, height= 8, aspect= 1)
            g.map(sns.boxplot, "species", col, orient="v", palette="Set2", order= xx)
            g.map(sns.stripplot, "species", col, jitter=True, color="grey", order= xx)
            #adjust_box_widths(g, 0.9)
            for ax in g.axes.flat:
                ax.set_xlabel(ax.get_xlabel(), fontsize=12)
                #ax.set_title(ax.get_title(),  fontsize=20)
                ax.set_ylabel(ax.get_ylabel(), fontsize=12)
                ax.tick_params(axis="x", labelsize=10)
                ax.tick_params(axis="y", labelsize=10)

            plt.savefig(col+"species_boxplot"+'.pdf', transparent=True)




    def boxplot2(self):
        df= self.table
        df.drop(["remark"], axis= 1)
        xx= sorted(df['sex'].unique(), reverse= True)
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        df1 = df.select_dtypes(include=numerics)
        for i, col in enumerate(df1.columns):

            g= sns.FacetGrid(df, col="species", sharex=False, sharey=False, height=4)
            g.map(sns.boxplot, "sex", col, orient="v", palette="Set2", order= xx)
            g.map(sns.stripplot, "sex", col, jitter=True, color=".25", order= xx)
            adjust_box_widths(g, 0.9)
            for ax in g.axes.flat:
                ax.set_xlabel(ax.get_xlabel(), fontsize=12)
                ax.set_title(ax.get_title(),  fontsize=12)
                ax.set_ylabel(ax.get_ylabel(), fontsize=12)
            plt.savefig(col+"sex_paired_boxplot"+'.pdf', transparent=True)

    def pcaplot(self):
        import sklearn.decomposition
        import sklearn.preprocessing
        colors= sns.color_palette("Dark2", 9)
        df= self.table
        df= df.dropna()
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        df1 = df.select_dtypes(include=numerics)
        df1 = sklearn.preprocessing.StandardScaler().fit_transform(df1)
        pca = sklearn.decomposition.PCA(n_components=2)
        projected = pca.fit_transform(df1)
        snp_pca = pd.DataFrame(projected,
                columns=['PC' + str(x) for x in range(1, 2+1)],
                index=df.index)
        snp_pca['species'] = df['species']
        keys = snp_pca['species'].unique()
        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel('Principal Component 1', fontsize = 15)
        ax.set_ylabel('Principal Component 2', fontsize = 15)
        ax.set_title('2 component PCA', fontsize = 20)
        targets = keys
        markers=["*", "o", "s", "D"]

        for target, color, marker in zip(targets, colors, markers):
            indicesToKeep = snp_pca['species'] == target
            ax.scatter(snp_pca.loc[indicesToKeep, 'PC1']
                       , snp_pca.loc[indicesToKeep, 'PC2'], c = color, marker= marker
                       , s = 50)
            ax.tick_params(axis="x", labelsize=15)
            ax.tick_params(axis="y", labelsize=15)
            ax.set_xlabel(ax.get_xlabel(),   fontsize=18)
            ax.set_title(ax.get_title(), fontsize=20)
            ax.set_ylabel(ax.get_ylabel(), fontsize=18)
            ax.legend(targets, loc= "best", prop=dict(size=15))
        plt.savefig("pca_plot"+'.pdf', transparent=True)



    def plsplot(self):
        import sklearn.preprocessing
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        df= self.table
        df= df.dropna()
        y= df['species']
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        df1 = df.select_dtypes(include=numerics)
        keys = df['species'].unique()
        colors= sns.color_palette("Dark2", 9)
        markers=["*", "o", "s", "D"]
        lda = LinearDiscriminantAnalysis(n_components=2)
        projected = lda.fit(df1, df['species']).transform(df1)
        snp_lda = pd.DataFrame(projected,
                 columns=['lda' + str(x) for x in range(1, 2+1)],
                 index=df.index)
        snp_lda['species'] = df['species']
        xx= list(range(len(df['species'])))
        keys = df['species'].unique()
        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(1,1,1)

        targets = keys
        markers=["*", "o", "s", "D"]
        for target, color, marker in zip(targets, colors, markers):

            indicesToKeep = snp_lda['species'] == target
            ax.scatter(snp_lda.loc[indicesToKeep, 'lda1']
                       , snp_lda.loc[indicesToKeep, 'lda2']
                       , c = color, marker= marker
                       , s = 50)
            ax.tick_params(axis="x", labelsize=15)
            ax.tick_params(axis="y", labelsize=15)
            ax.set_xlabel(ax.get_xlabel(),   fontsize=18)
            ax.set_title(ax.get_title(), fontsize=20)
            ax.set_ylabel(ax.get_ylabel(), fontsize=18)
            ax.legend(targets, loc= "best", prop=dict(size=15))
            ax.set_xlabel('Pls 1', fontsize = 15)
            ax.set_ylabel('pls 2', fontsize = 15)
            ax.set_title('2 Component LDA Plot', fontsize = 20)

        plt.savefig("lda_plot"+'.pdf', transparent=True)





if __name__ == "__main__":
    processplot("samplefile_morphometricanalyzer_extended.tab")
