# DeepRibo

DeepRibo is a deep neural network created by Clauwaert. J et. al. for the prediction of Open Reading Frames (ORF) in prokaryotes using ribosome profiling data. The package is written in python using the PyTorch library.

# Installation

To use DeepRibo, simply clone this repository in your working directory and install the necessary libraries:

`conda env create -f environment.yml
` 

# User Guide

`DataParser.py` and `DeepRibo.py` are the main scripts from which all functions can be executed. For more information about these functions, use the  `-h` flag.

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

The function will create two files for each ORF present in te genome. `*_seq.npy` is the binary image of 30 nucleotide covering the Shine-Dalgarno region  [-20,10]. `*_signal.npy` contains a vector with the riboseq coverage signal for each ORF. Depending on the label (CDS annotated in the gtf/gff), these will be listed under `<dest>/0` or `<dest>/1`.`<dest>/data_list.csv` contains a list of all samples with metadata. This file will be read and processed by the custom data loader used by `DeepRibo.py`. The parsed data of multiple datasets should all be present in one folder, according to this structure:

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




# Code Examples
These code examples will work with the data present if executed sequentially

### parsing the data
Parsing *E. coli*, *B. subtilis* and *S. typhimurium* data:

`python DataParser.py data/raw/eco_cov_sense.bedgraph data/raw/eco_cov_asense.bedgraph data/raw/eco_elo_sense.bedgraph data/raw/eco_elo_asense.bedgraph data/raw/eco.fa data/raw/eco.gtf data/processed/ecoli/`


`python DataParser.py data/raw/bac_cov_sense.bedgraph data/raw/bac_cov_asense.bedgraph data/raw/bac_elo_sense.bedgraph data/raw/bac_elo_asense.bedgraph data/raw/bac.fa data/raw/bac.gtf data/processed/bacillus`

`python DataParser.py data/raw/sal_cov_sense.bedgraph data/raw/sal_cov_asense.bedgraph data/raw/sal_elo_sense.bedgraph data/raw/sal_elo_asense.bedgraph data/raw/sal.fa data/raw/sal.gtf data/processed/salmonella/
`

### Training a model

`python DeepRibo.py train -data_path data/processed -train_data ecoli salmonella -test_data bacillus --cr 0.0 0.0 --cc 0.0 0.0 --tcr 0.0 --tcc 0.0 -dest models/my_model.pt -b 16 --GPU`

### Predicting with a model

`python DeepRibo.py predict -data_path data/processed -test_data bacillus --tcr 0 --tcc 0 -model model/my_model.pt -dest data/processed/bacillus/my_model_bac_pred.csv --GPU`

The output file is an extension of the `data_list.csv` file created when parsing the data. For more information about each column check the **Supplementary Information and Figures** from the [Online Article](.)

