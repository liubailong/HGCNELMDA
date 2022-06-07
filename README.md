### Computational method using Heterogeneous Graph Convolutional Network Model Combined with Reinforcement Layer for MiRNA-disease association prediction(HGCNELMDA)




### Implemented evironment

Python>=3.6

### Required libraries

`numpy,numba,pylab,time,openpyxl,xlrd,tensorflow,torch,sys,os,importlib,metrics,networkx`

We recommended that you could install Anaconda to meet these requirements


### How to run HGCNELMDA? 

#### Data

All data or mid results are organized in `DATA` fold, which contains miRNA-disease associations,disease semantic similarity, miRNA functional similarity, encode result of disease and  miRNA.

#### The starting point for running HGCNELMDA is:

(1)**main.py**：generating meta-paths from the dataset of miRNA-disease associations, disease semantic similarity,  miRNA functional similarity. all the result is saved in the folds named `"5.mid result"` and `"6.meta path"`, which need to be created by yourselves.

(2)**train.py**: training the model of HGCNELMDA, which will refer `utils.py, models.py,metrics.py. `
And it outputs the parameter of HGCNELMDA.

#### other relative files:

**models.py**: an auto_encoder model used in HGCNELMDA
**metrics.py**: functions of  metrics operating in HGCNELMDA
**utils.py**: useful functions in HGCNELMDA
