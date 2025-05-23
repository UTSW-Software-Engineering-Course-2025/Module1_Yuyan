"""
Usage:
  tsne.py [--input=<input_file>] [--dims=<int>] [--tol=<float>] [--perp=<float>] [--ini-momentum=<float>] [--final-momentum=<float>]
          [--stepsize=<float>] [--min-gain=<float>] [--iters=<int>] [--output=<output_file>] [--plot] [--save-fig]

Options:
  -h --help             Show this screen.
  --input=<input_file>       Input txt file [default: mnist2500_X.txt].
  --dims=<int>               Output dimensions [default: 2].
  --tol=<float>              Tolerance for beta search [default: 1e-5].
  --perp=<float>             Perplexity [default: 30.0].
  --ini-momentum=<float>    Initial momentum [default: 0.5].
  --final-momentum=<float>   Final momentum [default: 0.8].
  --stepsize=<float>         Learning rate [default: 500].
  --min-gain=<float>         Minimum gain value [default: 0.01].
  --iters=<int>              Number of iterations [default: 1000].
  --output=<output_file>     Output file path [default: tsne_result.npy].
  --plot                     Whether to show plot.
  --save-fig                 Whether to save the plot.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from adjustbeta import adjustbeta
from docopt import docopt
import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
def pca(X, no_dims=50):
    """
    Runs PCA on the nxd array X in order to reduce its dimensionality to
    no_dims dimensions.

    Parameters
    ----------
    X : numpy.ndarray
        data input array with dimension (n,d)
    no_dims : int
        number of dimensions that PCA reduce to

    Returns
    -------
    Y : numpy.ndarray
        low-dimensional representation of input X
    """
    n, d = X.shape
    X = X - X.mean(axis=0)[None, :]
    _, M = np.linalg.eig(np.dot(X.T, X))
    Y = np.real(np.dot(X, M[:, :no_dims]))
    return Y




def dis_matrix (X):
    """
    Compute pairwise squared Euclidean distance matrix.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data.

    Returns
    -------
    D : ndarray of shape (n_samples, n_samples)
        Distance matrix.
    """
    Xi_2 = np.sum(X**2, axis=1).reshape(-1, 1)  
    D = Xi_2 - 2 * np.dot(X, X.T) + Xi_2.T 
    return D

def compute_p_ij(P):
    """
    Symmetrize conditional probability matrix P.

    Parameters
    ----------
    P : ndarray
        Conditional probabilities.

    Returns
    -------
    p_ij : ndarray
        Symmetric joint probabilities.
    """
    num = P + P.T
    den = np.sum(P+P.T)
    p_ij = num / den
    return p_ij

def compute_q_ij(Y):
    """
    Compute low-dimensional joint probabilities q_ij from Y.

    Parameters
    ----------
    Y : ndarray of shape (n_samples, n_components)
        Low-dimensional embeddings.

    Returns
    -------
    q_ij : ndarray
        Joint probabilities in low-dimensional space.
    """
    distance_y =  dis_matrix(Y)
    num = (1/(1 + distance_y))
    den = (np.sum(num))
    q_ij = num / den 
    return q_ij

def compute_kl(p_ij, q_ij):
    """
    Compute the Kullback-Leibler (KL) divergence between two probability matrices.

    Parameters
    ----------
    P : ndarray of shape (n_samples, n_samples)
        High-dimensional joint probability matrix.
    Q : ndarray of shape (n_samples, n_samples)
        Low-dimensional joint probability matrix.

    Returns
    -------
    C : float
        The KL divergence value KL(P || Q).
    """
    p_ij = np.maximum(p_ij, 1e-12)  # avoid log(0)
    q_ij = np.maximum(q_ij, 1e-12)
    return np.sum(p_ij * np.log(p_ij / q_ij))

def compute_dY(p_ij, q_ij, Y):
    """
    Compute gradient of the KL divergence with respect to Y.

    Parameters
    ----------
    p_ij : ndarray
        High-dimensional joint probabilities.
    q_ij : ndarray
        Low-dimensional joint probabilities.
    Y : ndarray
        Current low-dimensional embeddings.

    Returns
    -------
    dY : ndarray
        Gradient with respect to Y.
    """
    diff = (p_ij - q_ij )
    inverse = (1/(1+dis_matrix(Y)))
    diff_y = Y[:, None, :] - Y[None, :, :]
    dY = np.sum(((diff*inverse)[:,:,None])*diff_y,1)
    del diff, inverse, diff_y
    return dY


def tsne(X,no_dims=2, tol = 1e-5, perplexity =30.0, ini_momentum=0.5, final_momentum= 0.8, stepsize =500, min_gain = 0.01, T = 1000, plot= True, save_fig = False):
    """
    Perform t-SNE on high-dimensional data X.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        High-dimensional input data.
    no_dims : int
        Output dimensionality (usually 2 or 3).
    tol : float
        Tolerance for binary search in P computation.
    perplexity : float
        Perplexity used to balance local/global aspects of the embedding.
    ini_momentum : float
        Initial momentum value used in early optimization.
    final_momentum : float
        Final momentum value used after `early exaggeration` phase.
    stepsize : float
        Learning rate (step size for gradient descent).
    min_gain : float
        Minimum gain applied during gradient update scaling.
    T : int
        Number of optimization iterations.
    plot : bool
        Whether to display the resulting embedding as a 2D or 3D plot.
    save_fig : bool
        Whether to save the plot as a static image or HTML.

    Returns
    -------
    Y : ndarray of shape (n_samples, no_dims)
        Low-dimensional embedding of the data.

    Raises
    ------
    ValueError
        If `X` is not a 2D array.
        If `perplexity` is greater than the number of samples.
        If `no_dims` is not 2 or 3 when `plot=True`.
        If `stepsize`, `perplexity`, or `T` are non-positive.

    Examples
    --------
    >>> from tsne import tsne
    >>> X = np.random.rand(100, 50)
    >>> Y = tsne(X, no_dims=2, perplexity=30.0, T=1000, plot=True, save_fig=False)
    >>> Y.shape
    (100, 2)
    """
    n= X.shape [0]

    # calculate P_j|i by X 
    P, beta = adjustbeta(X, tol, perplexity)

    Y = pca( X)
    Y = Y[:,:no_dims]
    # initialize delta_Y and gains 
    delta_Y = np.zeros((n,no_dims))
    gains = np.ones((n,no_dims))
    p_ij = compute_p_ij(P)*4
    
    for t in tqdm(range (T)):
        # compute q_ij from Y 
        q_ij = compute_q_ij(Y) 
        np.fill_diagonal(q_ij,0)
        # compute gradient of Y 
        dY = compute_dY(p_ij, q_ij, Y)
        # choose momentum to be initial momentum if t<20, otherwise momentum is final_momentum 
        if t < 20 :
            momentum = ini_momentum
        else :
            momentum = final_momentum
        
        # compute and update gains
        gains = (gains + 0.2)* ((dY > 0)!= (delta_Y>0))+ (gains*0.8)*((dY>0 )==(delta_Y>0))
        
        #clip it to be at least min_gain
        gains = np.maximum(min_gain, gains)

        # update delta_Y and Y 
        delta_Y = momentum*delta_Y - stepsize*( gains * dY)
        Y = Y + delta_Y

        # remove exaggeration 
        if t == 100 :
            p_ij = p_ij/4
        
    return Y 

if __name__ == "__main__":
    args = docopt(__doc__)
    # Load data from file
    X = np.loadtxt(args["--input"])
    X = pca(X, 50)
    labels = np.loadtxt("mnist2500_labels.txt")
    Y = tsne(
        X,
        no_dims=int(args["--dims"]),
        tol=float(args["--tol"]),
        perplexity=float(args["--perp"]),
        ini_momentum=float(args["--ini-momentum"]),
        final_momentum=float(args["--final-momentum"]),
        stepsize=float(args["--stepsize"]),
        min_gain=float(args["--min-gain"]),
        T=int(args["--iters"]),
        plot=args["--plot"],
        save_fig=args["--save-fig"]
    )
    # draw 2d scatter plot
    if int(args["--dims"]) == 2:
        plt.scatter(Y[:, 0], Y[:, 1], 20, labels)
        plt.savefig("mnist_tsne.png")
    # draw 3d scatter plot 
    if int(args["--dims"]) == 3:
        print("3D t-SNE selected")
        labels = labels.astype(int).astype(str) 
        df = pd.DataFrame({
            "x": Y[:, 0],
            "y": Y[:, 1],
            "z": Y[:, 2],
            "label": labels
        })
        # Create plot
        fig = px.scatter_3d(df, x="x", y="y", z="z",  color="label", color_discrete_sequence=px.colors.qualitative.G10)
        # Create Dash app
        app = dash.Dash(__name__)
        app.layout = html.Div([
            html.H3("Simple 3D t-SNE Visualization"),
            dcc.Graph(figure=fig)
        ])

        app.run(debug=True)


    