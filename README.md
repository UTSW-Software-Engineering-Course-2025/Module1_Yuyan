# Dimensionality Reduction: GraphDR & t-SNE

This repository provides implementations of **GraphDR** and **t-SNE** for effective visualization of high-dimensional data such as single-cell RNA-seq or PCA-reduced features. These methods help uncover intrinsic structures by projecting complex data into 2D or 3D space.

- **GraphDR**: A graph-based method that smooths data with Laplacian regularization before projecting it with or without rotation using eigendecomposition.
- **t-SNE**: A stochastic neighbor embedding algorithm widely used to capture local structure and visualize data clusters.

---

##  Installation
### Clone the repository 

```bash
git clone https://github.com/UTSW-Software-Engineering-Course-2025/Module1_Yuyan.git
cd Module1_Yuyan
```
### requirement 
- Python 3.8+
- numpy
- scipy
- matplotlib
- seaborn
- scikit-learn
#### or directly install all dependencies via 
```bash
conda env create -f environment.yml
conda activate tsne_graphdr

```
##   quickstart 

### tsne 
```bash
cd tsne # cd tsne folder 
python tsne.py # run with default setting 
```
or full parameters see 
```bash
python tsne.py -h
```

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


### graphdr 

```bash
cd GraphDr # cd tsne folder 
python graphdr.py # run with default setting 
```
or full parameters see 
```bash
python graphdr.py -h
```
GraphDR & PCA Preprocessing CLI Tool.

Usage:
  graphdr.py [--data=<path>] [--anno=<path>] [--lambda=<lambda>] [--neighbors=<int>] [--no-rotation] [--no-plot]
  graphdr.py (-h | --help)

Options:
  -h --help             Show this screen.  
  --data=<path>         Path to gene expression matrix [default: hochgerner_2018.data.gz].  
  --anno=<path>         Path to annotation/label file [default: hochgerner_2018.anno].
  --lambda=<lambda>     Laplacian regularization strength [default: 20.0].  
  --neighbors=<int>     Number of nearest neighbors for graph [default: 10].  
  --no-rotation         Disable eigenvector rotation.  
  --no-plot             Disable saving projection plot.  
 




