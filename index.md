# GEO Signature Extractor

`GESgnExt` is a gene expression signature extractor on GEO studies. It can be used to construct the signatures for the subject (e.g. gene, drug, disease) in each study. The signatures can be further used to build the similarity network for inferring the associations among those subjects. `GESgnExt` consists of multiple components for signature construction. It provides a set of functions to evaluate the performance of different methods for each component. The most recommended model for the classification components is `UDT-RF` and the one for the clustering component is `Kallima`. The trained models can be adopted for predictions of subject categorization and control/perturbation sample discrimination. It also provides several utility functions to manipulate the dataset and post-process the results.

## Getting Started

The following instructions will help you get a copy of the source code as well as the datasets, and run the programs on your own machine.

### Prerequisities

Firstly, you need to install a Python Interpreter (tested 2.7.12) and these packages:

* numpy (tested 1.13.1)
* scipy (tested 0.19.1)
* matplotlib (tested 2.0.2)
* pandas (tested 0.20.3)
* scikit-learn (tested 0.19.0)
* optunity (tested 1.1.1)
* binarytree (tested 2.0.1)
* pyyaml (test 3.12)
* openpyxl (test 2.4.8)
* pymemcache \[optional\] \(tested 1.4.3\)
* rdflib \[optional\] \(tested 4.2.2\)
* becas \[optional\] \(tested 1.0.3\)
* apiclient \[optional\] \(tested 1.0.3\)
* graphviz \[optional\] \(tested 2.38.0\)
* r-essentials \[optional\] \(tested 1.6.0\)
* rpy2 \[optional\] \(tested 2.8.5\)

The simplest way to get started is to use [Anaconda](https://www.continuum.io/anaconda-overview) Python distribution. If you have limited disk space, the [Miniconda](http://conda.pydata.org/miniconda.html) installer is recommended. After installing Miniconda and adding the path of folder `bin` to `$PATH` variable, run the following command:

```bash
conda install scikit-learn pandas matplotlib optunity openpyxl
```

### Download the Source Code

You can clone the repository of this project and then update the submodule after entering the main folder:

```bash
git clone https://github.com/cskyan/gesgnext.git
cd gesgnext
git submodule update --init --recursive
```

Or you can clone the repository and submodules simultaneously:

```bash
git clone --recursive https://github.com/cskyan/gesgnext.git
```

### Configure Environment Variable

* Add the path of folder `bin` to `$PATH` variable so that you can run the scripts wherever you want. *Remember to grant execution permissions to all the files located in* `bin`
* Add the path of folder `lib` to `$PYTHONPATH` variable so that the Python Interpreter can find the library `bionlp`.

### Configuration File

The global configuration file is stored as `etc/config.yaml`. The configurations of different functions in different modules are separated, which looks like the code snippet below.

```
MODULE1:
- function: FUNCTION1
  params:
    PARAMETER1: VALUE1
    PARAMETER2: VALUE2
- function: FUNCTION2
  params:
    PARAMETER1: VALUE1
	
MODULE2:
- function: FUNCTION1
  params:
    PARAMETER1: VALUE1
```

Hence you can access a specific parameter VALUE using a triple (MODULE, FUNCTION, PARAMETER). The utility function `cfg_reader` in `bionlp.util.io` can be used to read the parameters in the configuration file:

```python
import bionlp.util.io as io
cfgr = io.cfg_reader(CONFIG_FILE_PATH)
cfg = cfgr(MODULE, FUNCTION)
VALUE = cfg[PARAMETER]
```

The parameters under the function `init` means that they are defined in module scope, while the parameters under the function `common` means that they are shared among all the functions inside the corresponding module.

### Locate the Pre-Generated Dataset

After cloning the repository, you can download some pre-generated datasets [here](https://data.mendeley.com/datasets/y7gnb79gfb) . The datasets described below are organized as [csr sparse matrices](http://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html), stored in compressed `npz` files using the [function](http://docs.scipy.org/doc/numpy/reference/generated/numpy.savez_compressed.html) of `numpy`. 

Filename | Description  
--- | ---
signatures.zip | Constructed Signatures
gse_doc.pkl | Series documents
gsm_doc.pkl | Sample documents
orig_gse_X.npz | Annotated Series dataset
orig_gsm_X.npz | Annotated Sample dataset
orig_gsm_X_[0-2].npz | Annotated Sample dataset for [disease\|drug\|gene]
gsm2gse.npz | Sample-Series index mapping
creeds_gse_X.npz | Annotated Series dataset derived using CREEDS
creeds_gsm_X_[0-2].npz | Annotated Sample dataset for [disease\|drug\|gene] derived using CREEDS
udt_gse_X.npz | Annotated Series dataset derived using UDT
udt_gsm_X_[0-2].npz | Annotated Sample dataset for [disease\|drug\|gene] derived using UDT
gse_Y.npz | Subject type labels
gsm_Y.npz | Control/perturbation sample labels
gsm_y_[0-2].npz | Control/perturbation sample labels for [disease\|drug\|gene]
gsm_lb_[0-2].npz | Sample group labels for [disease\|drug\|gene]
circos_cache.tar.gz | Circos plot cache
demo_signatures.zip | Demo constructed signatures
demo_gse_doc.pkl | Demo Series documents
demo_gsm_doc.pkl | Demo Sample documents
demo_gse_X.npz | Demo annotated Series dataset
demo_gsm_X.npz | Demo annotated Sample dataset
demo_gsm_X_[0-2].npz | Demo annotated Sample dataset for [disease\|drug\|gene]
demo_gsm_y_[0-2].npz | Demo control/perturbation sample labels for [disease\|drug\|gene]
demo_gsm_lb_[0-2].npz | Demo sample group labels for [disease\|drug\|gene]
demo_data.tar.gz | Demo sample data
demo_gedata.tar.gz | Demo Gene Expression Data
demo_dge.tar.gz | Demo Differential Gene Expression Data
demo_simmt.tar.gz | Demo similarity network
demo_circos_cache.tar.gz | Demo circos plot cache


**In order to locate the dataset you want to use, please remove the prefixes of 'orig_', 'creeds_', and 'udt_'. And change the parameter `DATA_PATH` of module `bionlp.spider.gsc` inside `etc/config.yaml` into the location of 'X.npz'.**

You can load a dataset into a [Pandas DataFrame](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html), with the corresponding `GEO Accession Number` as index (e.g. GSEXXX, GSMXXX) and each feature as column name, using the utility function `read_df` in `bionlp.util.io`:

```python
import bionlp.util.io as io
X = io.read_df('X.npz', with_idx=True, sparse_fmt='csr')
```

### A Simple Example

You can run a demo using the following command:

```bash
gsx_extrc.py -m demo
```

*If your operating system is Windows, please use the Python Interpreter to execute the python scripts:*

```bash
python gsx_extrc.py -m demo
```

This demo will automatically download a pre-generated dataset and perform a 5-fold cross validation on each classification component of `GESgnExt`, followed by the signature construction. The log is printed to standard output and the results are saved on the disk.

## Run a specific component
`GESgnExt` mainly has four components, `gse_clf` (Subject Categorization), `gsm_clf` (Control/Perturbation Sample Discrimination), `gsm_clt` (Sample Clustering), `gen_sgn` (Signature Generation). You can run each component using parameter `-m` to indicate the component name. For example:

```bash
gsx_extrc.py -m gsm_clf
```

You can also run them together by not setting the parameter `-m`. 

## Parameter Tuning

For the sake of the best performance, you should tune the parameters of your selected model and write them on the model configuration file so that you can use these tuned parameters for model evaluation. 

### Setup parameter range

You can edit the function `gen_mdl_params` inside `bin/gsx_extrc.py` to change the range of parameter tuning. Please uncomment the code lines corresponded to your selected model and change the range of the parameters or append other values you want to test.

### Run parameter tuning script

You can choose an approach for parameter tuning using the following command.

*Particle Swarm Search*:

```bash
gsx_extrc.py -t -r particle_swarm
```

*Grid Search*:

```bash
gsx_extrc.py -t -r grid search
```

*Random Search*:

```bash
gsx_extrc.py -t -r random search
```

More details about the search methods (solvers) please refer to the documents of [Optunity](http://optunity.readthedocs.io/en/latest/user/solvers.html)

### Covert the result to configuration file

You can use the utility function in `bin/gsx_helper.py` to transformat your tuning result by the following command:

```bash
gsx_helper.py -m n2y -l TUNING_OUTPUT_FOLDER_PATH
```

**Then copy the basename of the configuration file ended with `.yaml` to the parameter `mdl_cfg` of module `gsx_extrc` inside `etc/config.yaml`.**

The pre-tuned parameters for some models are stored in `etc/mdlcfg.yaml`.

## Model Evaluation

You can use different combination of the feature selection model and classification model to generate a pipeline as the final computational model.

You can uncomment the corresponding code lines of the models you want to evaluate in functions `gen_featfilt`, `gen_clfs`, and `gen_clt_models` inside `bin/gsx_extrc.py` for feature selection and classification respectively.

In addition, you can use command line parameter `-c` to adopt the pre-combined model in functions `gen_cb_models` and `gen_cbclt_models`. To make use of the parameters stored in configuration file, you can use command line parameter `-c -b` to adopt the pre-combined model with optimized parameters.

## Dataset Re-Generation

You can re-generate the dataset from the [annotated signatures](http://amp.pharm.mssm.edu/creeds) stored in `DATA_PATH` using the following command:

```bash
gsx_gendata.py -m gen
```

It will also generate separated label data `gsm_y_[0-2].npz` and `gsm_lb_[0-2].npz` for single label running.

Feature selection method can also be applied to the dataset in advance by uncommenting the corresponding code line in function `gen_data` inside `bin\gsx_gendata.py`.

If you only want to apply feature selection on the generated dataset or generate separated label data, you can use command line parameter `-l`. Make sure your dataset has already been renamed as 'gse_X.npz' or 'gsm_X_[0-2].npz', and the processed dataset will be generated as '#method#+#num_of_features#\_[gse\|gsm]\_X[_[0-2]].npz'.

## Common Parameter Setting

* _-p [0-2]_  
specify which label you want to use independently
* _-l_  
indicate that you want to use all labels simultaneously
* _-k NUM_  
specify *K*-fold cross validation
* _-a [micro | macro]_  
specify which average strategy you want to use for multi-label annotation
* _-n NUM_  
specify how many CPU cores you want to use simultaneously for parallel computing

**Other parameter specification can be obtained using `-h`.**
