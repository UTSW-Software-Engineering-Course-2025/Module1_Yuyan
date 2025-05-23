"""
GraphDR & PCA Preprocessing CLI Tool.

Usage:
  graphdr.py [--data=<path>] [--anno=<path>] [--lambda=<lambda>] [--neighbors=<int>] [--no-rotation] [--no-plot]
  graphdr.py (-h | --help)

Options:
  -h --help             Show this screen.
  --data=<path>         Path to gene expression matrix [default: hochgerner_2018.data.gz].
  --anno=<path>         Path to annotation/label file [default: hochgerner_2018.anno]
  --lambda=<lambda>     Laplacian regularization strength [default: 20.0].
  --neighbors=<int>     Number of nearest neighbors for graph [default: 10].
  --no-rotation         Disable eigenvector rotation.
  --no-plot             Disable saving projection plot.
"""
from docopt import docopt
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import identity, csgraph
from sklearn.decomposition import PCA
from scipy.sparse.linalg import spsolve
import plotly


def data_preprocessing (data, plot_pca= True):
    """
    Parameters
    ----------
    data : pandas.DataFrame
        Gene expression matrix with shape (n_genes, n_cells), where rows are genes
        and columns are single cells.
    plot_pca : bool, default=True
        Whether to generate and save a scatter plot of the first two principal components
        as 'pca_data.png'.

    Returns
    -------
    preprocessed_data : ndarray of shape (n_cells, 20)
        PCA-reduced representation of the input expression matrix.
    """
    percell_sum = data.sum(axis=0)
    pergene_sum = data.sum(axis=1)

    preprocessed_data = data / percell_sum.values[None, :] * np.median(percell_sum)
    preprocessed_data = preprocessed_data.values

    #transform the preprocessed_data array by `x := log (1+x)`
    preprocessed_data = np.log(1 + preprocessed_data)

    #standard scaling
    preprocessed_data_mean = preprocessed_data.mean(axis=1)
    preprocessed_data_std = preprocessed_data.std(axis=1)
    preprocessed_data = (preprocessed_data - preprocessed_data_mean[:, None]) / \
                        preprocessed_data_std[:, None]

    #pca dimension reduction
    pca_tool = PCA(n_components = 20)
    pca_tool.fit(preprocessed_data.T)

    # shape (n,d)
    preprocessed_data = pca_tool.transform(preprocessed_data.T)
    
    if plot_pca:
        plt.figure(figsize=(15, 10))
        seaborn.scatterplot(x=preprocessed_data[:, 0], y=preprocessed_data[:, 1], linewidth=0, s=5, hue=anno)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PCA Projection')
        plt.tight_layout()
        plt.savefig('pca_data.png', dpi=300)
        plt.close()

    return preprocessed_data


import numpy as np
import matplotlib.pyplot as plt
import seaborn
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csgraph

def graphdr(pca_data, anno, lambda_=20.0, no_rotation=True, num_neighbors=10, plot_graphdr=True):
    """
    Perform GraphDR dimensionality reduction using graph Laplacian smoothing.

    Parameters
    ----------
    pca_data : ndarray of shape (n_samples, n_features)
        Input data matrix in PCA-reduced space.
    anno : array-like of shape (n_samples,)
        Labels or annotations used for coloring the scatter plot.
    lambda_ : float, default=20.0
        Regularization strength for Laplacian smoothing.
    no_rotation : bool, default=False
        If True, skips eigenvector rotation.
    num_neighbors : int, default=10
        Number of neighbors to build the graph.
    plot_graphdr : bool, default=True
        If True, saves a scatter plot of the result.

    Returns
    -------
    Z : ndarray of shape (n_samples, n_features) or (n_samples, 2)
        The transformed low-dimensional embedding.

    Raises
    ------
    ValueError
        If `pca_data` has fewer rows than `num_neighbors`.

    See Also
    --------
    numpy.linalg.eigh : Computes eigenvalues and eigenvectors of a symmetric matrix.
    sklearn.neighbors.kneighbors_graph : Constructs the k-nearest neighbor graph.
    scipy.sparse.csgraph.laplacian : Computes the Laplacian matrix of a graph.


    Examples
    --------
    >>> Z = graphdr(pca_data, 10, True, 10)
    >>> Z.shape
    (10000, 2)

    """
    n, p = pca_data.shape

    # Construct identity matrix and k-NN graph
    I = np.identity(n)
    A = kneighbors_graph(pca_data, n_neighbors=num_neighbors, mode='connectivity').toarray()

    # Compute Laplacian and smoothing operator
    L = csgraph.laplacian(A, symmetrized=True)
    inverse_L = np.linalg.inv(I + lambda_ * L)

    # Compute projection matrix or skip rotation
    if not no_rotation:
        mul = pca_data.T @ inverse_L @ pca_data
        _, eigvec = np.linalg.eigh(mul)
        Z = inverse_L @ pca_data @ eigvec
    else:
        Z = inverse_L @ pca_data
        
    # Optional: generate plot
    if plot_graphdr:
        plt.figure(figsize=(15, 10))
        seaborn.scatterplot(x=Z[:, 0], y=Z[:, 1], linewidth=0, s=3, hue=anno)
        plt.xlabel('GraphDR 1')
        plt.ylabel('GraphDR 2')
        plt.title('GraphDR Projection')
        plt.tight_layout()
        plt.savefig(f'graphdr_result_{num_neighbors}nn_{lambda_}.png', dpi=300)
        plt.close()

    return Z


if __name__ == "__main__":
    # read in data 
    args = docopt(__doc__)
    data = pd.read_csv(args['--data'],sep='\t',index_col=0)
    anno = pd.read_csv(args['--anno'],sep='\t',header=None)
    anno = anno[1].values
    pca_data =  data_preprocessing (data)
    graphdr_result = graphdr(pca_data,anno,20,True,num_neighbors=10)
