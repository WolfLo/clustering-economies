import pandas as pd
import numpy as np
from collections import defaultdict

from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn import metrics
import hdbscan

from scipy.cluster import hierarchy
from fastcluster import linkage

from fancyimpute import KNN

import matplotlib.pyplot as plt

import seaborn as sns

# %matplotlib inline
# plt.style.use('seaborn-white')


class Preprocessing:
    def __init__(self, csv_path, varlist=None, verbose=False):
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
            varlist = ['Country Name'] + varlist
            self.df = self.df[varlist]
        # convert all columns but Country Names to numeric type
        self.df.iloc[:, 1:] = \
            self.df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
        # report poor features and selected_countries
        if verbose:
            feature_miss = self.df.isnull().sum()
            country_miss = self.df.isnull().sum(axis=1)
            feature_miss = \
                feature_miss[feature_miss != 0].sort_values(ascending=False)
            country_miss = \
                country_miss[country_miss != 0].sort_values(ascending=False)
            print('MISSING VALUES FOR EACH FEATURE:')
            print(feature_miss, '\n')
            print('MISSING VALUES FOR EACH COUNTRY:')
            print(country_miss)

    # def drop_poor_columns(self, p):
    #     ''' Drop the columns of self.df with more than p (%) missing values'''
    #
    #     # create df with a the count of missing values for each column
    #     missing_df = pd.DataFrame(self.df.isnull().sum())
    #     # extract the names of columns with more than p (%) missing values
    #     poor_columns = missing_df.loc[missing_df[0] > p*len(self.df)].index
    #     # drop sparse columns
    #     self.df.drop(poor_columns, axis=1, inplace=True)
    #     return self.df, poor_columns

    def dropPoorFeatures(self, axis, p):
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

    def imputeKNN(self):
        # df is my data frame with the missings. I keep only floats
        self.country_names = self.df['Country Name'].values
        df_numeric = self.df.select_dtypes(include=[np.float64]).values
        # impute missing values
        df_filled_KNN = pd.DataFrame(
            KNN(k=2, verbose=False).complete(df_numeric))
        df_filled_KNN.insert(
            loc=0, column='Country Names', value=self.country_names)
        df_filled_KNN.columns = self.df.columns
        df_filled_KNN.index = self.df.index
        return df_filled_KNN

    def exportCSV(self, path, impute=False):
        if not impute:
            # export the cleaned dataframe to a csv file
            self.df.to_csv(path)
        else:
            # impute the missing values before exporting to csv
            self.df_filled_KNN = self.imputeKNN()
            self.df_filled_KNN.to_csv(path)


def heatmap(df, links):
    '''
    Plot a matrix dataset as a hierarchically-clustered heatmap,
    using given linkages.
    '''
    cmap = sns.cubehelix_palette(
        as_cmap=True, start=.5, rot=-.75, light=.9)
    sns.clustermap(
        data=df, row_linkage=links, col_cluster=False, cmap=cmap)


class Clustering:
    def __init__(self, csv_path, verbose=False):
        self.df = pd.read_csv(csv_path)
        # change index (row labels)
        self.df = self.df.set_index('Country Code', verify_integrity=True)
        # df.info(verbose=False)
        # store country full names (for plots) before removing the feature
        self.country_names = self.df['Country Name'].values
        self.df = self.df.drop(['Country Name'], axis=1)
        # scale the dataset to be distributed as a standard Gaussian
        cols = self.df.columns
        ind = self.df.index
        self.df = pd.DataFrame(scale(self.df))
        self.df.columns = cols
        self.df.index = ind
        # create disctionary of clusters
        self.clusterings = defaultdict(lambda: np.array(0))
        self.clusterings_labels = defaultdict(lambda: np.array(0))
        # print general info
        if verbose:
            print('The imported dataset as the following characteristics:')
            print(self.df.info(verbose=False))

    def getPC(self):
        '''
        Calculate the principal components (PC) and create a new DataFrame
        by projecting the datapoints on the PC space.
        '''
        self.pca = PCA()
        self.pca_loadings = pd.DataFrame(
            PCA().fit(self.df).components_.T, index=self.df.columns)
        self.df_pc = pd.DataFrame(
            self.pca.fit_transform(self.df), index=self.df.index)

        # plot the cumulated proportion of variance explained by the PC
        print('CUMULATIVE PROPORTION OF VARIANCE EXPLAINED BY PCs')
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

    def plotAlongPC(self, pc1=0, pc2=1, xlim=[-5, 5], ylim=[-5, 5],
                    loadings=True, clustering=None):
        '''
        Plot the countries along the two principal components given in input:
        pc1[int] (usually = 0, indicating the first PC) and pc2[int]
        '''
        fig, ax1 = plt.subplots(figsize=(9, 7))

        ax1.set_xlim(xlim[0], xlim[1])
        ax1.set_ylim(ylim[0], ylim[1])

        if clustering is not None:
            # build a generator of colors
            NUM_COLORS = len(self.clusterings[clustering])
            clist = np.random.uniform(low=0, high=1, size=(NUM_COLORS, 4))
            # plot countries along PCs coloring them according to their cluster
            labels = self.clusterings_labels[clustering]
            for i, country in enumerate(self.df_pc.index):
                ax1.annotate(country,
                             (self.df_pc[pc1].loc[country],
                              -self.df_pc[pc2].loc[country]),
                             ha='center',
                             color=clist[labels[i]],
                             fontweight='bold')
        else:
            # plot countries along PCs
            for i in self.df_pc.index:
                ax1.annotate(i,
                             (self.df_pc[pc1].loc[i],
                              -self.df_pc[pc2].loc[i]),
                             ha='center',
                             color='b',
                             fontweight='bold')

        # Plot reference lines
        ax1.hlines(0, -5, 5, linestyles='dotted', colors='grey')
        ax1.vlines(0, -5, 5, linestyles='dotted', colors='grey')
        pc1_string = 'Principal Component ' + str(pc1)
        pc2_string = 'Principal Component ' + str(pc2)
        ax1.set_xlabel(pc1_string)
        ax1.set_ylabel(pc2_string)

        if loadings:
            # Plot Principal Component loading vectors, using a second y-axis.
            ax2 = ax1.twinx().twiny()

            ax2.set_ylim(-1, 1)
            ax2.set_xlim(-1, 1)
            ax2.tick_params(axis='y', colors='orange')
            # ax2.set_xlabel('Principal Component loading vectors',
            # color='orange')

            # Plot labels for vectors.
            # 'a' is an offset parameter to separate arrow tip and text.
            a = 1.07
            for i in self.pca_loadings[[pc1, pc2]].index:
                ax2.annotate(i,
                             (self.pca_loadings[pc1].loc[i]*a,
                              -self.pca_loadings[pc2].loc[i]*a),
                             color='orange')

            # Plot vectors
            for k in range(len(self.pca_loadings.columns)):
                ax2.arrow(0, 0, self.pca_loadings[pc1][k],
                          -self.pca_loadings[pc2][k],
                          width=0.002, color='black')
        return

    def plotDendrogram(self, links, threshold, metric, method):
        plt.figure(figsize=(15, 9))
        den_title = 'METHOD: ' + str(method) + ' METRIC: ' + str(metric)
        plt.title(den_title)
        den = hierarchy.dendrogram(links,
                                   orientation='right',
                                   labels=self.country_names,
                                   color_threshold=threshold,
                                   leaf_font_size=10)
        plt.vlines(threshold, 0,
                   plt.gca().yaxis.get_data_interval()[1],
                   colors='r', linestyles='dashed')

        return den

    def clustersTable(self, clustering):
        '''
        Clustering is an array of cluster labels, one for each country
        '''
        lis = sorted(
            list(zip(clustering, self.country_names)), key=lambda x: x[0])
        groups = set(map(lambda x: x[0], lis))
        table = pd.DataFrame(list(
            zip(groups, [[y[1] for y in lis if y[0] == x] for x in groups])))
        table.columns = ['Cluster', '']
        table.set_index('Cluster', inplace=True, verify_integrity=False)
        return table

    def saveClustering(self, cluster_labels, clustering_name):
        # save clusterings into a dict and rename its columns
        self.clusterings[clustering_name] = \
            self.clustersTable(cluster_labels)
        self.clusterings[clustering_name].columns = [clustering_name]
        self.clusterings_labels[clustering_name] = cluster_labels

    def hierarchicalClustering(
            self, metric, method, threshold=None, on_PC=0, heatmap=False):
        '''
        Show figures of clusters retrieved through the hierachical method
        and return an array with the cluster index of each country.

        metric -- [str] used for assigning distances to data:
                   'euclidean', 'ćorrelation', 'cosine', 'seuclidean'...
        method -- [str] the type of linkage used for agglomerating the nodes
                    'average','complete','ward'...(check fastcluster full list)
        threshold -- [int] threshold distance for separing clusters,
                     in the hierachical tree.
        on_PC -- [int] apply clustering by using data projections
                 on the first on_PC principal components
        '''
        if on_PC > 0:
            df = self.df_pc.iloc[:, :on_PC+1]
        else:
            df = self.df

        if method == 'all':
            method = ['average',
                      'complete',
                      'single',
                      'weighted',
                      'centroid',  # only for Euclidean data
                      'median',  # only for Euclidean data
                      'ward',  # only for Euclidean data
                      ]
        elif type(method) != list:
            method = list([method])
        metric = str(metric)

        for met in method:
            # set up the linking tool
            links = linkage(df, metric=metric, method=met)
            self.link = links
            # plot dendrogram
            self.plotDendrogram(links, threshold, metric, met)
            if heatmap:
                heatmap(df, links)

            labels = hierarchy.fcluster(links, threshold, criterion='distance')
            # save clusters
            self.saveClustering(
                labels, 'hc_'+str(met)+'_'+str(metric)+'_'+str(threshold))

        # self.hierarchical_classes = get_hierarchical_classes(den)
        # plt.savefig('tree2.png')

    def hdbscan(self, min_cluster_size=2, on_PC=0):
        '''compute clusters using HDBSCAN algorithm'''
        if on_PC > 0:
            df = self.df_pc.iloc[:, :on_PC+1]
        else:
            df = self.df
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        clusterer.fit_predict(df)
        # save clusters
        self.saveClustering(clusterer.labels_, 'hdbscan')

    def bayesianGaussianMixture(self, n_components, covariance_type='full',
                                n_init=50, on_PC=0):
        '''
        Compute Bayesian Gaussian Mixture clustering.
        Note: in this case, the number of components effectively used
        can be < n_componentss (at most, n_components).
        '''
        if on_PC > 0:
            df = self.df_pc.iloc[:, :on_PC+1]
        else:
            df = self.df
        clusterer = BayesianGaussianMixture(n_components,
                                            covariance_type=covariance_type,
                                            n_init=n_init)
        labels = clusterer.fit(df).predict(df)
        # save clusters
        self.saveClustering(labels, 'bayesian gm' + str(n_components))

    def gaussianMixture(self, n_components, covariance_type='full',
                        n_init=50, on_PC=0):
        '''compute Gaussian Mixture clustering'''
        if on_PC > 0:
            df = self.df_pc.iloc[:, :on_PC+1]
        else:
            df = self.df
        clusterer = GaussianMixture(n_components,
                                    covariance_type=covariance_type,
                                    n_init=n_init)
        labels = clusterer.fit(df).predict(df)
        # save clusters
        self.saveClustering(labels, 'gm' + str(n_components))

    def gmBIC(self, n_min, n_max, covariance_type='full',
              n_init=50, on_PC=0):
        if on_PC > 0:
            df = self.df_pc.iloc[:, :on_PC+1]
        else:
            df = self.df
        '''compute Bayesian Information Criterion'''
        n_components = np.arange(n_min, n_max)
        models = [
            GaussianMixture(n, covariance_type=covariance_type, n_init=n_init)
            for n in n_components]
        bics = [model.fit(df).bic(df) for model in models]
        bics = np.array(bics)
        # store the optimal number of gaussian components and the resulting BIC
        self.min_BIC = [bics.argmin()+n_min, bics.min()]
        print('the minimum BIC is achieved with \
              %i gaussian components' % self.min_BIC[0])
        fig, ax = plt.subplots(num='Bayesian Information Criterion')
        plt.plot(n_components, bics)

    def kmeans(self, n_clusters=2, on_PC=0, n_init=50, evaluate=True):
        '''compute clusters using KMeans algorithm'''
        if on_PC > 0:
            df = self.df_pc.iloc[:, :on_PC+1]
        else:
            df = self.df
        # re-initialize seed for random initial centroids' position
        np.random.seed(42)
        clusterer = KMeans(n_clusters=n_clusters, n_init=n_init)
        clusterer.fit_predict(df)
        # save clusters
        self.saveClustering(clusterer.labels_, 'kmeans' + str(n_clusters))
        # compute Silhouette and Calinski-Harabaz Score
        if evaluate:
            benchClustering(clusterer, 'kmeans', df)

    def multipleKmeans(self, k_min, k_max, on_PC=0, n_init=50):
        if on_PC > 0:
            df = self.df_pc.iloc[:, :on_PC+1]
        else:
            df = self.df

        ks = np.arange(k_min, k_max)
        silh = np.zeros(k_max - k_min)
        cal_har = np.zeros(k_max - k_min)
        for k in ks:
            # re-initialize seed for random initial centroids' position
            np.random.seed(42)
            clusterer = KMeans(n_clusters=k, n_init=n_init)
            clusterer.fit_predict(df)
            silh[k-k_min] = metrics.silhouette_score(
                df, clusterer.labels_, metric='euclidean')
            cal_har[k-k_min] = metrics.calinski_harabaz_score(
                df, clusterer.labels_)

        # multiple line plot
        fig, ax1 = plt.subplots(num='How many clusters?')
        color = 'green'
        ax1.set_xlabel('Number of clusters')
        ax1.set_ylabel('Silhouette Score', color=color)
        plt.plot(ks, silh, marker='o', markerfacecolor=color,
                 markersize=6, color=color, linewidth=2)

        ax2 = ax1.twinx()
        color = 'orange'
        ax2.set_ylabel('Calinski-Harabaz Score', color=color)
        plt.plot(ks, cal_har, marker='o', markerfacecolor=color,
                 markersize=6, color=color, linewidth=2)
        ax1.grid(True)
        plt.legend()
        return silh, cal_har

    def country_links(self, clustering_array):
        # given a clustering, build a table of country links
        ll = len(clustering_array)
        tab = np.zeros((ll, ll))
        for k in range(ll):
            tab[k] = (clustering_array == clustering_array[k])
        return tab

    def clustering_similarities(self):
        n_methods = len(self.clusterings_labels)
        n_countries = len(self.country_names)
        size = n_countries**2
        tab = np.zeros((n_methods, n_countries, n_countries))
        sim = np.zeros((n_methods, n_methods))
        # for each clustering, build the table of country links
        for k, clus in enumerate(self.clusterings_labels):
            tab[k] = self.country_links(self.clusterings_labels[clus])
        # for each clustering's table of links,
        # calculate its similarity to the others
        for k in range(n_methods):
            for kk in range(n_methods):
                sim[k][kk] = (tab[k] == tab[kk]).sum()/size
                methods = list(self.clusterings_labels.keys())
        sim = pd.DataFrame(sim, index=methods, columns=methods)
        return sim


def plotBarh(df, by_column):
    '''
    Horizontal bar chart with by_column value for each country.
    by_column - column name of the variable to plot as bars [str]
    '''

    newdf = df.sort_values(by=str(by_column))
    x = np.array(newdf[by_column])
    y = np.array(newdf['Country Name'])
    y_pos = np.arange(len(y))

    fig, ax = plt.subplots(figsize=(7, 10))

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y)
    ax.invert_yaxis()  # labels read top-to-bottom
    # ax.set_xlabel('%GDP')
    ax.set_title(by_column)

    ax.barh(y_pos, x, color='b')


def plotMultiBarh(df, by_columns, country_names):
    '''
    #TODO fix this
    Horizontal bar chart with by_columns value for each country.
    by_columns - list of variables to plot as bars
    '''
    newdf = df.sort_values(by=by_columns)
    x = np.array(newdf[by_columns])
    y = np.array(country_names)
    y_pos = np.arange(len(y))

    fig, ax = plt.subplots(figsize=(7, 10))

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y)
    ax.invert_yaxis()  # labels read top-to-bottom
    # ax.set_xlabel('%GDP')
    # ax.set_title(by_column)
    bar_width = 0.2
    width = 0.2
    colors = list('rgby')
    for i in range(x.shape[1]):
        ax.barh(y_pos + width, x[:, i], bar_width,
                color=colors[i], label=by_columns[i])
        width += width

    ax.legend()


def benchClustering(estimator, name, data):
    silh = metrics.silhouette_score(
        data, estimator.labels_, metric='euclidean')
    cal_har = metrics.calinski_harabaz_score(data, estimator.labels_)
    return silh, cal_har


def highlight_max(data, color='yellow'):
    '''
    highlight the maximum in a Series or DataFrame
    '''
    attr = 'background-color: {}'.format(color)
    # remove % and cast to float
    data = data.replace('%', '', regex=True).astype(float)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index=data.index, columns=data.columns)
