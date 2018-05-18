# DeepRibo

DeepRibo is a deep neural network created by Clauwaert. J et. al. for the prediction of Open Reading Frames (ORF) in prokaryotes using ribosome profiling data. The package is written in python using the PyTorch library. This repository contains the code necessary to train your own models. However, the weights of the six models discussed in the [Article](.) are given in `models/` and can therefore be directly used as a tool to make predictions. It is stronly recommended to use GPU's for the use of DeepRibo.

# Installation

To use DeepRibo, simply clone this repository in your working directory and install the necessary libraries:

	git clone https://github.com/Biobix/DeepRibo.git
	conda env create -f environment.yml
 

# User Guide

`src/DataParser.py` and `src/DeepRibo.py` are the main scripts from which all functions can be executed. For more information about these functions, use the  `-h` flag.

## Parsing Data
First, data has to be converted into the required format. `DataParser.py` takes care of this with minimal effort. Several files are required to successfully parse the data:

- **[sense_cov]**    Path to .bedgraph containing sense riboseq data (coverage)
- **[asense_cov]**   Path to .bedgraph containing antisense riboseq data (coverage)
- **[sense_elo]**    Path to .bedgraph containing sense riboseq data (elongating)
- **[asense_elo]**   Path to .bedgraph containing antisense riboseq data (elongating)
- **[fasta]**        Path to .fasta/.fa containing genome sequence
- **[gtf]**      Path to .gtf/.gff containing annotation
- **[dest]**  Path to output destination. This path must contain two folders
               named 0 and 1

The function will create two files for each ORF present in te genome. `*_seq.npy` is the binary image of 30 nucleotide covering the Shine-Dalgarno region  [-20,10]. `*_reads.npy` contains a vector with the riboseq coverage signal for each ORF. For each ORFs present in the gff/gtf file (feature column annotated as CDS) a positive label is attributed. All samples will accordingly be be listed under `<dest>/0` (negative label) or `<dest>/1` (positive label).`<dest>/data_list.csv` contains a list of all samples with metadata. This file will be read and processed by the custom data loader used by `DeepRibo.py`. The parsed data of multiple datasets should all be present in one folder, according to the following structure:

------------
    DATA
    ├── ecoli
    │   ├── 0
    │   │   ├ ...
    │   ├── 1
    │   │   ├ ...
    │   ├── data_list.csv
    ├── bacillus
    │   ├── 0
    │   │   ├ ...
    │   ├── 1
    │   │   ├ ...
    │   ├── data_list.csv
    ├── ...

----


# Training a model

After all data has been processed, a model can be trained. Any or all datasets present in the **DATA** folder can be used for training/testing. For more information about the function parameters simply use

`python DeepRibo.py train -h`

During training, the model's weights are saved after each epoch. A json object is furthermore created after training containing all of the performance metrics for the training/testing data. These performance metrics include the loss, acceracy, Area Under the Roc Curve (AUC-ROC) and Area under the Precision-Recall Curve (PR-AUC). When training the model, cut-off values for both the training data and test data, based upon the minimal coverage and RPKM values of each sample, are given to filter out data with low to non-existent signal (see [Full Article](.)). two cut-off values have to be given for each dataset used for training/testing, and are required to be in the same order. To obtain the right values for each dataset, an R script is provided which uses the SiZer package. The function `get_cutoff_values` is listed in `src/s_curve_cutoff_estimation.R`. Simply run the script with the required parameters to obtain these values after parsing the data.

	# start R	
	R
	# load the functions from the script
	>source('s_curve_cutoff_estimation.R')
	# list the dataset and the path to which the png figure is stored
	>get_cutoff_values(path="../../data/processed/<your data>/data_list.csv", dest="figure")
	$min_RPKM
	  ....
	$min_coverage
       	  ....

 
# Making predictions 

Once a model has been trained it can be used to make predictions on any other data you have parsed. For more information about the required parameters simply use the help flag:

``python DeepRibo.py predict -h`

The output file is an extension of the `data_list.csv` file created when parsing the data. For more information about each column check the **Supplementary Information and Figures** from the [Online Article](.)

# Pretrained models

All six models discussed in the [Full Article](.) are located in `models/`, and can be directly used for making predictions on personal data. User data has to be first parsed using `DataParser.py`. Before any predictions can be made, cut-off values have to be determined on the user data using the R-script `src/s_curve_cutoff_estimation.R`.


# Code Examples
These code examples will work with the data present if executed sequentially

### parsing the data
Parsing *E. coli*, *B. subtilis* and *S. typhimurium* data:

`python DataParser.py ../data/raw/eco_cov_sense.bedgraph ../data/raw/eco_cov_asense.bedgraph ../data/raw/eco_elo_sense.bedgraph ../data/raw/eco_elo_asense.bedgraph ../data/raw/eco.fa ../data/raw/eco.gtf ../data/processed/ecoli`

`python DataParser.py ../data/raw/bac_cov_sense.bedgraph ../data/raw/bac_cov_asense.bedgraph ../data/raw/bac_elo_sense.bedgraph ../data/raw/bac_elo_asense.bedgraph ../data/raw/bac.fa ../data/raw/bac.gtf ../data/processed/bacillus`

`python DataParser.py ../data/raw/sal_cov_sense.bedgraph ../data/raw/sal_cov_asense.bedgraph ../data/raw/sal_elo_sense.bedgraph ../data/raw/sal_elo_asense.bedgraph ../data/raw/sal.fa ../data/raw/sal.gtf ../data/processed/salmonella`

### Training a model

`python DeepRibo.py train ../data/processed --train_data ecoli salmonella --test_data bacillus --tr_rpkm 0.0 0.0 --tr_cov 0.0 0.0 --te_rpkm 0.0 --te_cov 0.0 --dest models/my_model.pt -b 16 --GPU`

**DISCLAIMER** : Normally the cut-off values are not going to be 0. Including data with zero signal in the ribosome profiling data will create a bad model 
### Predicting with a model

`python DeepRibo.py predict ../data/processed --pred_data bacillus -pr 0 -pc 0 --model ../models/my_model_5.pt --dest ../data/processed/bacillus/my_model_bac_pred.csv`
