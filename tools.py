import pandas as pd
import numpy as np

from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

from fancyimpute import KNN

import matplotlib.pyplot as plt


class Preprocessing:
    def __init__(self, csv_path, varlist=None):
        '''
        path -- the string of the csv file representing our raw dataset
        varlist -- the list of strings
        '''
        # import the csv dataset as a pandas DataFrame
        self.df = pd.read_csv(csv_path)
        # change index (row labels)
        self.df = self.df.set_index('Country Code', verify_integrity=True)
        # only keep the variables(columns) selected by user
        if varlist:
            self.df = self.df[varlist]
        # convert all columns but Country Names to numeric type
        self.df.iloc[:, 1:] = \
            self.df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
        # delete bad poor_columns
        # delete bad rows

    def drop_poor_columns(self, p):
        ''' Drop the columns of self.df with more than p (%) missing values'''

        # create df with a the count of missing values for each column
        missing_df = pd.DataFrame(self.df.isnull().sum())
        # extract the names of columns with more than p (%) missing values
        poor_columns = missing_df.loc[missing_df[0] > p*len(self.df)].index
        # drop sparse columns
        self.df.drop(poor_columns, axis=1, inplace=True)
        return self.df, poor_columns

    def drop_poor_features(self, axis, p):
        '''
        Drop the rows/columns of self.df with more than p (%) missing values
        axis -- indicate whether to drop rows (axis=0) or columns(axis=1)
        '''
        # create df with the count of missing values for each row/column
        missing_df = pd.DataFrame(self.df.isnull().sum(axis=int(not axis)))
        # extract the names of rows/columns with more than p (%) missing values
        if axis == 0:
            length = len(self.df.columns)
        else:
            length = len(self.df)
        poor_features = missing_df.loc[missing_df[0] > p*length].index
        # drop sparse rows/columns
        self.df.drop(poor_features, axis=axis, inplace=True)
        return self.df, poor_features

    def impute_KNN(self):
        # df is my data frame with the missings. I keep only floats
        df_numeric = self.df.select_dtypes(include=[np.float64]).as_matrix()
        # impute missing values
        df_filled_KNN = pd.DataFrame(KNN(3).complete(df_numeric))
        df_filled_KNN.columns = self.df.columns
        df_filled_KNN.index = self.df.index
        return df_filled_KNN

    def export_csv(self, path, impute=False):
        if not impute:
            # export the cleaned dataframe to a csv file
            self.df.to_csv(path)
        else:
            # impute the missing values before exporting to csv
            df_filled_KNN = self.impute_KNN()
            df_filled_KNN.to_csv(path)


class PCAnalysis:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        # change index (row labels)
        self.df = self.df.set_index('Country Code', verify_integrity=True)
        # df.info(verbose=False)
        # scale the dataset to be distributed as a standard Gaussian
        cols = self.df.columns
        ind = self.df.index
        self.df = pd.DataFrame(scale(self.df))
        self.df.columns = cols
        self.df.index = ind
        # print general info
        print('The imported dataset as the following characteristics:')
        self.df.info(verbose=False)

    def get_PC(self):
        '''
        Calculate the principal components (PC) and create a new DataFrame
        by projecting the datapoints on the PC space.
        '''
        self.pca = PCA()
        self.df_pc = pd.DataFrame(
            self.pca.fit_transform(self.df), index=self.df.index)

        # plot the cumulated proportion of variance explained by the PC
        plt.figure(figsize=(7, 5))

        plt.plot(range(1, len(self.pca.components_)+1),
                 self.pca.explained_variance_ratio_, '-o',
                 label='Individual component')
        plt.plot(range(1, len(self.pca.components_)+1),
                 np.cumsum(self.pca.explained_variance_ratio_), '-s',
                 label='Cumulative')

        plt.ylabel('Proportion of Variance Explained')
        plt.xlabel('Principal Component')
        plt.xlim(0.75, 4.25)
        plt.ylim(0, 1.05)
        plt.xticks(range(1, len(self.pca.components_)+1))
        plt.legend(loc=2)

        return self.df_pc

    def plot_along_PC(self, pc1=0, pc2=1):
        '''
        Plot the countries along the two principal components given in input:
        pc1[int] (usually = 0, indicating the first PC) and pc2[int]
        '''
        fig, ax1 = plt.subplots(figsize=(9, 7))

        ax1.set_xlim(-5, 5)
        ax1.set_ylim(-5, 5)

        # Plot Principal Components pc1 and pc2
        for i in self.df_pc.index:
            ax1.annotate(i,
                         (self.df_pc[pc1].loc[i], -self.df_pc[pc2].loc[i]),
                         ha='center')

        # Plot reference lines
        ax1.hlines(0, -5, 5, linestyles='dotted', colors='grey')
        ax1.vlines(0, -5, 5, linestyles='dotted', colors='grey')
        pc1_string = 'Principal component' + str(pc1)
        pc2_string = 'Principal component' + str(pc2)
        ax1.set_xlabel(pc1_string)
        ax1.set_ylabel(pc2_string)
        return
