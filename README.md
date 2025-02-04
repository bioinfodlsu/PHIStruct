# PHIStruct (Phage-Host Interaction Prediction with Structure-Aware Protein Embeddings)

![badge][badge-jupyter]
![badge][badge-python]
![badge][badge-pandas]
![badge][badge-numpy]
![badge][badge-scipy]
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white) <br>
[![Actions Status](https://github.com/bioinfodlsu/PHIStruct/workflows/Check%20for%20syntax%20errors/badge.svg)](https://github.com/bioinfodlsu/PHIStruct/actions)
[![Actions Status](https://github.com/bioinfodlsu/PHIStruct/workflows/Run%20Black%20formatter/badge.svg)](https://github.com/bioinfodlsu/PHIStruct/actions)
![badge][badge-github-actions]

**PHIStruct** is a phage-host interaction prediction tool that uses structure-aware protein embeddings to represent the receptor-binding proteins (RBPs) of phages. By incorporating structure information, it presents improvements over using sequence-only protein embeddings and feature-engineered sequence properties &mdash; especially for phages with RBPs that have low sequence similarity to those of known phages.

**Paper**: https://doi.org/10.1093/bioinformatics/btaf016

If you find our work useful, please consider citing:

```
@article{10.1093/bioinformatics/btaf016,
    author = {Gonzales, Mark Edward M and Ureta, Jennifer C and Shrestha, Anish M S},
    title = {PHIStruct: Improving phage-host interaction prediction at low sequence similarity settings using structure-aware protein embeddings},
    journal = {Bioinformatics},
    pages = {btaf016},
    year = {2025},
    month = {01},
    issn = {1367-4811},
    doi = {10.1093/bioinformatics/btaf016},
    url = {https://doi.org/10.1093/bioinformatics/btaf016}
}
```

You can also find PHIStruct on [bio.tools](https://bio.tools/phistruct).

## Table of Contents

- [📰 News](https://github.com/bioinfodlsu/PHIStruct?tab=readme-ov-file#-news)
- [♾️ Run on Google Colab](https://github.com/bioinfodlsu/PHIStruct?tab=readme-ov-file#%EF%B8%8F-run-on-google-colab)
- [🚀 Installation & Usage](https://github.com/bioinfodlsu/PHIStruct?tab=readme-ov-file#-installation--usage)
- [📚 Description](https://github.com/bioinfodlsu/PHIStruct?tab=readme-ov-file#-description)
- [🔬 Dataset of Predicted Structures of Receptor-Binding Proteins](https://github.com/bioinfodlsu/PHIStruct?tab=readme-ov-file#-dataset-of-predicted-structures-of-receptor-binding-proteins)
- [🧪 Reproducing Our Results](https://github.com/bioinfodlsu/PHIStruct?tab=readme-ov-file#-reproducing-our-results)
- [💻 Authors](https://github.com/bioinfodlsu/PHIStruct?tab=readme-ov-file#-authors)

## 📰 News

- **13 Jan 2025** - Our [paper](https://doi.org/10.1093/bioinformatics/btaf016) is now published in _**Bioinformatics**_.
- **06 Nov 2024** - We presented our work at the **2024 Australian Bioinformatics and Computational Biology Society (ABACBS) National Conference** in Sydney. Poster [here](https://drive.google.com/file/d/1_IEL9WibxHFAjiMW-UmLa2X7G5Kvq4WQ/view?usp=sharing).

## ♾️ Run on Google Colab

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://phistruct.bioinfodlsu.com)

**You can readily run PHIStruct on Google Colab, without the need to install anything on your own computer: [http://phistruct.bioinfodlsu.com](http://phistruct.bioinfodlsu.com)**

## 🚀 Installation & Usage

**Operating System**: Windows (using [WSL](https://learn.microsoft.com/en-us/windows/wsl/install)), Linux, or macOS

Clone the repository:

```
git clone https://github.com/bioinfodlsu/PHIStruct
cd PHIStruct
```

Create a virtual environment with the dependencies installed via Conda (we recommend using [Miniconda](https://docs.anaconda.com/free/miniconda/index.html)):

```
conda env create -f environment.yaml
```

Activate this environment by running:

```
conda activate PHIStruct
```

Depending on your operating system, run the correct installation command (refer to the **last column** of the table below) to install and configure the remaining dependencies (you only need to do this once, that is, at installation):

| OS/Build          | Command for Checking OS/Build             | Installation Command |
| ----------------- | ----------------------------------------- | -------------------- |
| Linux AVX2 Build  | `cat /proc/cpuinfo \| grep avx2`          | `bash init.sh avx2`  |
| Linux SSE2 Build  | `cat /proc/cpuinfo \| grep sse2`          | `bash init.sh sse2`  |
| Linux ARM64 Build | `dpkg --print-architecture` or `uname -m` | `bash init.sh arm64` |
| macOS             | &ndash;                                   | `bash init.sh osx`   |

**Note**: Running the `init.sh` script may take a few minutes since it involves downloading a model (SaProt, around 5 GB) from Hugging Face.

### Running PHIStruct

```
python3 phistruct.py --input <input_dir> --model <model_joblib> --output <results_dir>
```

- Replace `<input_dir>` with the path to the directory storing the PDB files describing the structures of the receptor-binding proteins. Sample PDB files are provided [here](https://github.com/bioinfodlsu/PHIStruct/tree/main/sample_pdb).
- Replace `<model_joblib>` with the path to the trained model (recognized format: joblib or compressed joblib, framework: scikit-learn). Download our trained model from this [link](https://drive.google.com/file/d/1hf2UDs0rt34_T6FaUc5nB7g_kQLWFoi_/view?usp=sharing). No need to uncompress, but doing so will speed up loading the model albeit at the cost of additional storage requirements. Refer to this [guide](https://joblib.readthedocs.io/en/latest/generated/joblib.dump.html) for the list of accepted compressed formats.
- Replace `<results_dir>` with the path to the directory to which the results of running PHIStruct will be written. The results of running PHIStruct on the sample PDB files are provided [here](https://github.com/bioinfodlsu/PHIStruct/tree/main/sample_results).

The results for each protein are written to a CSV file (without a header row). Each row contains two comma-separated values: a host genus and the corresponding prediction score (class probability). The rows are sorted in order of decreasing prediction score. Hence, the first row pertains to the top-ranked prediction.

Under the hood, this script first converts each protein into a structure-aware protein embedding using SaProt and then passes the embedding to a multilayer perceptron trained on _all_ the entries in our dataset with host among the ESKAPEE genera ([link](https://drive.google.com/file/d/17yxaoeCF8H_rBIGPibP9qJUlL_N7H8kt/view?usp=sharing)). If your machine has a GPU, it will automatically be used to accelerate the protein embedding generation step.

### Training PHIStruct

```
python3 train.py --input <training_dataset>
```

- Replace `<training_dataset>` with the path to the training dataset. A sample can be downloaded [here](https://drive.google.com/file/d/17yxaoeCF8H_rBIGPibP9qJUlL_N7H8kt/view?usp=sharing).

The training dataset should be formatted as a CSV file (without a header row) where each row corresponds to a training sample. The first column is for the protein IDs, the second column is for the host genera, and the next 1,280 columns are for the components of the SaProt embeddings.

This script will output a gzip-compressed, serialized version of the trained model with filename `phistruct_trained.joblib.gz`.

↑ _Return to [Table of Contents](https://github.com/bioinfodlsu/PHIStruct?tab=readme-ov-file#table-of-contents)._

## 📚 Description

**Motivation:** Recent computational approaches for predicting phage-host interaction have explored the use of sequence-only protein language models to produce embeddings of phage proteins without manual feature engineering. However, these embeddings do not directly capture protein structure information and structure-informed signals related to host specificity.

**Method:** We present PHIStruct, a multilayer perceptron that takes in structure-aware embeddings of receptor-binding proteins, generated via the structure-aware protein language model SaProt, and then predicts the host from among the ESKAPEE genera.

**Results:** Compared against recent tools, PHIStruct exhibits the best balance of precision and recall, with the highest and most stable F1 score across a wide range of confidence thresholds and sequence similarity settings. The margin in performance is most pronounced when the sequence similarity between the training and test sets drops below 40%, wherein, at a relatively high-confidence threshold of above 50%, PHIStruct presents a 7% to 9% increase in class-averaged F1 over machine learning tools that do not directly incorporate structure information, as well as a 5% to 6% increase over BLASTp.
<br><br>
<img src="https://github.com/bioinfodlsu/PHIStruct/blob/main/figure.png?raw=True" alt="Teaser Figure" width = 800>
<br><br>
↑ _Return to [Table of Contents](https://github.com/bioinfodlsu/PHIStruct?tab=readme-ov-file#table-of-contents)._

## 🔬 Dataset of Predicted Structures of Receptor-Binding Proteins

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11202338.svg)](https://doi.org/10.5281/zenodo.11202338)

We also release a [dataset of protein structures](https://doi.org/10.5281/zenodo.11202338), computationally predicted via [ColabFold](https://doi.org/10.1038/s41592-022-01488-1), of 19,081 non-redundant (i.e., with duplicates removed) receptor-binding proteins from 8,525 phages across 238 host genera. We identified these receptor-binding proteins based on GenBank annotations. For phage sequences without GenBank annotations, we employed a pipeline that uses the viral protein library [PHROG](https://doi.org/10.1093/nargab/lqab067) and the machine learning model [PhageRBPdetect](https://doi.org/10.3390/v14061329).

↑ _Return to [Table of Contents](https://github.com/bioinfodlsu/PHIStruct?tab=readme-ov-file#table-of-contents)._

## 🧪 Reproducing Our Results

### Project Structure

The [`experiments`](https://github.com/bioinfodlsu/PHIStruct/tree/main/experiments) folder contains the files and scripts for reproducing our results. Note that additional (large) files have to be downloaded (or generated) following the instructions in the Jupyter notebooks.

<details>
  <summary>Click here to show/hide the list of directories, Jupyter notebooks, and Python scripts, as well as the folder structure.</summary>

#### Directories

| Directory                                                                                         | Description                                                                                                                                                                                                                                                                                                                        |
| ------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`data`](https://github.com/bioinfodlsu/PHIStruct/tree/main/experiments/inphared)                 | Contains the data (including the FASTA files and embeddings)                                                                                                                                                                                                                                                                       |
| [`preprocessing`](https://github.com/bioinfodlsu/PHIStruct/tree/main/experiments/preprocessing)   | Contains text files related to the preprocessing of host information and the identification of annotated receptor-binding proteins                                                                                                                                                                                                 |
| [`rbp_prediction`](https://github.com/bioinfodlsu/PHIStruct/tree/main/experiments/rbp_prediction) | Contains the trained model [PhageRBPdetect](https://www.mdpi.com/1999-4915/14/6/1329) (in JSON format), used for the computational prediction of receptor-binding proteins. Downloaded from this [repository](https://github.com/dimiboeckaerts/PhageRBPdetection/blob/main/data/RBPdetect_xgb_model.json) (under the MIT License) |
| [`temp`](https://github.com/bioinfodlsu/PHIStruct/tree/main/experiments/temp)                     | Contains intermediate output files during preprocessing, exploratory data analysis, and performance evaluation                                                                                                                                                                                                                     |

#### Jupyter Notebooks

| Notebook                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Description                                                                                       |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| [`1. Sequence Preprocessing.ipynb`](https://github.com/bioinfodlsu/PHIStruct/blob/main/experiments/1.%20Sequence%20Preprocessing.ipynb)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Preprocessing of host information and identification of annotated receptor-binding proteins       |
| [`2. RBP Computational Prediction.ipynb`](https://github.com/bioinfodlsu/PHIStruct/blob/main/experiments/2.%20RBP%20Computational%20Prediction.ipynb)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Computational prediction of receptor-binding proteins                                             |
| [`3.0. Data Consolidation (SaProt).ipynb`](<https://github.com/bioinfodlsu/PHIStruct/blob/main/experiments/3.0.%20Data%20Consolidation%20(SaProt).ipynb>) <br> [`3.1. Data Consolidation (ProstT5 - AA Tokens).ipynb`](<https://github.com/bioinfodlsu/PHIStruct/blob/main/experiments/3.1.%20Data%20Consolidation%20(ProstT5%20-%20AA%20tokens).ipynb>) <br> [`3.2. Data Consolidation (PST).ipynb`](<https://github.com/bioinfodlsu/PHIStruct/blob/main/experiments/3.2.%20Data%20Consolidation%20(PST).ipynb>) <br> [`3.3. Data Consolidation (SaProt with Low-Confidence Masking).ipynb`](<https://github.com/bioinfodlsu/PHIStruct/blob/main/experiments/3.3.%20Data%20Consolidation%20(SaProt%20with%20Low-Confidence%20Masking).ipynb>) <br> [`3.4. Data Consolidation (SaProt with Structure Masking).ipynb`](<https://github.com/bioinfodlsu/PHIStruct/blob/main/experiments/3.4.%20Data%20Consolidation%20(SaProt%20with%20Structure%20Masking).ipynb>) <br> [`3.5. Data Consolidation (SaProt with Sequence Masking).ipynb`](<https://github.com/bioinfodlsu/PHIStruct/blob/main/experiments/3.5.%20Data%20Consolidation%20(SaProt%20with%20Sequence%20Masking).ipynb>) <br> [`3.6. Data Consolidation (ProstT5 - 3Di Tokens).ipynb`](<https://github.com/bioinfodlsu/PHIStruct/blob/main/experiments/3.6.%20Data%20Consolidation%20(ProstT5%20-%203Di%20Tokens).ipynb>)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | Generation of CSV files consolidating the proteins, phage-host information, and embeddings        |
| [`4. Exploratory Data Analysis.ipynb`](https://github.com/bioinfodlsu/PHIStruct/blob/main/experiments/4.%20Exploratory%20Data%20Analysis.ipynb)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Exploratory data analysis                                                                         |
| [`5.0. Classifier Building & Evaluation (SaProt).ipynb`](<https://github.com/bioinfodlsu/PHIStruct/blob/main/experiments/5.0.%20Classifier%20Building%20%26%20Evaluation%20(SaProt).ipynb>) <br> [`5.1. Benchmarking - Classifier Building & Evaluation (ProstT5 - AA Tokens).ipynb`](<https://github.com/bioinfodlsu/PHIStruct/blob/main/experiments/5.1.%20Benchmarking%20-%20Classifier%20Building%20%26%20Evaluation%20(ProstT5%20-%20AA%20Tokens).ipynb>) <br> [`5.2. Benchmarking - Classifier Building & Evaluation (PST).ipynb`](<https://github.com/bioinfodlsu/PHIStruct/blob/main/experiments/5.2.%20Benchmarking%20-%20Classifier%20Building%20%26%20Evaluation%20(PST).ipynb>) <br> [`5.3. Benchmarking - Classifier Building & Evaluation (ESM-1b).ipynb`](<https://github.com/bioinfodlsu/PHIStruct/blob/main/experiments/5.3.%20Benchmarking%20-%20Classifier%20Building%20%26%20Evaluation%20(ESM-1b).ipynb>) <br> [`5.4. Benchmarking - Classifier Building & Evaluation (ESM-2).ipynb`](<https://github.com/bioinfodlsu/PHIStruct/blob/main/experiments/5.4.%20Benchmarking%20-%20Classifier%20Building%20%26%20Evaluation%20(ESM-2).ipynb>) <br> [`5.5. Benchmarking - Classifier Building & Evaluation (ProtT5).ipynb`](<https://github.com/bioinfodlsu/PHIStruct/blob/main/experiments/5.5.%20Benchmarking%20-%20Classifier%20Building%20%26%20Evaluation%20(ProtT5).ipynb>) <br> [`5.6. Benchmarking - Classifier Building & Evaluation (SaProt with Low-Confidence Masking).ipynb`](<https://github.com/bioinfodlsu/PHIStruct/blob/main/experiments/5.6.%20Benchmarking%20-%20Classifier%20Building%20%26%20Evaluation%20(SaProt%20with%20Low-Confidence%20Masking).ipynb>) <br> [`5.7. Benchmarking - Classifier Building & Evaluation (SaProt with Structure Masking).ipynb`](<https://github.com/bioinfodlsu/PHIStruct/blob/main/experiments/5.7.%20Benchmarking%20-%20Classifier%20Building%20%26%20Evaluation%20(SaProt%20with%20Structure%20Masking).ipynb>) <br> [`5.8. Benchmarking - Classifier Building & Evaluation (SaProt with Sequence Masking).ipynb`](<https://github.com/bioinfodlsu/PHIStruct/blob/main/experiments/5.8.%20Benchmarking%20-%20Classifier%20Building%20%26%20Evaluation%20(SaProt%20with%20Sequence%20Masking).ipynb>) <br> [`5.9. Benchmarking - Classifier Building & Evaluation (ProstT5 - 3Di Tokens).ipynb`](<https://github.com/bioinfodlsu/PHIStruct/blob/main/experiments/5.9.%20Benchmarking%20-%20Classifier%20Building%20%26%20Evaluation%20(ProstT5%20-%203Di%20Tokens).ipynb>) <br> [`5.10. Benchmarking - Classifier Building & Evaluation (SeqVec).ipynb`](<https://github.com/bioinfodlsu/PHIStruct/blob/main/experiments/5.10.%20Benchmarking%20-%20Classifier%20Building%20%26%20Evaluation%20(SeqVec).ipynb>) <br> [`5.11. Benchmarking - Classifier Building & Evaluation (Random Forest).ipynb`](<https://github.com/bioinfodlsu/PHIStruct/blob/main/experiments/5.11.%20Benchmarking%20-%20Classifier%20Building%20%26%20Evaluation%20(Random%20Forest).ipynb>) <br> [`5.12. Benchmarking - Classifier Building & Evaluation (SVM).ipynb`](<https://github.com/bioinfodlsu/PHIStruct/blob/main/experiments/5.12.%20Benchmarking%20-%20Classifier%20Building%20%26%20Evaluation%20(SVM).ipynb>) | Construction of phage-host interaction prediction model, benchmarking, and performance evaluation |
| [`6.0. Comparison.ipynb`](https://github.com/bioinfodlsu/PHIStruct/blob/main/experiments/6.0.%20Comparison.ipynb) <br> [`6.0. Comparison - Weighted.ipynb`](https://github.com/bioinfodlsu/PHIStruct/blob/main/experiments/6.0.%20Comparison%20-%20Weighted.ipynb) <br> [`6.1. Plotting - F1.ipynb`](https://github.com/bioinfodlsu/PHIStruct/blob/main/experiments/6.1.%20Plotting%20-%20F1.ipynb) <br> [`6.1. Plotting - F1 - Weighted.ipynb`](https://github.com/bioinfodlsu/PHIStruct/blob/main/experiments/6.1.%20Plotting%20-%20F1%20-%20Weighted.ipynb) <br> [`6.2. Plotting - PR Curve.ipynb`](https://github.com/bioinfodlsu/PHIStruct/blob/main/experiments/6.2.%20Plotting%20-%20PR%20Curve.ipynb) <br> [`6.2. Plotting - PR Curve - Weighted.ipynb`](https://github.com/bioinfodlsu/PHIStruct/blob/main/experiments/6.2.%20Plotting%20-%20PR%20Curve%20-%20Weighted.ipynb) <br> [`6.3. Confusion Matrix.ipynb`](https://github.com/bioinfodlsu/PHIStruct/blob/main/experiments/6.3.%20Confusion%20Matrix.ipynb) <br>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Tabular and graphical comparison of the performance of our model versus benchmarks                |

#### Python Scripts

| Script                                                                                                            | Description                                                                                                                                                          |
| ----------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`ClassificationUtil.py`](https://github.com/bioinfodlsu/PHIStruct/blob/main/experiments/ClassificationUtil.py)   | Contains the utility functions for the constructing the training and test sets, building the phage-host interaction prediction model, and evaluating its performance |
| [`ConstantsUtil.py`](https://github.com/bioinfodlsu/PHIStruct/blob/main/experiments/ConstantsUtil.py)             | Contains the constants used in the notebooks and scripts                                                                                                             |
| [`MLPDropout.py`](https://github.com/bioinfodlsu/PHIStruct/blob/main/experiments/MLPDropout.py)                   | Implements a multilayer perceptron with dropout in scikit-learn                                                                                                      |
| [`RBPPredictionUtil.py`](https://github.com/bioinfodlsu/PHIStruct/blob/main/experiments/RBPPredictionUtil.py)     | Contains the utility functions for the computational prediction of receptor-binding proteins                                                                         |
| [`SequenceParsingUtil.py`](https://github.com/bioinfodlsu/PHIStruct/blob/main/experiments/SequenceParsingUtil.py) | Contains the utility functions for preprocessing host information and identifying annotated receptor-binding proteins                                                |
| [`StructureUtil.py`](https://github.com/bioinfodlsu/PHIStruct/blob/main/experiments/StructureUtil.py)             | Contains the utility functions for consolidating the embeddings generated via structure-aware protein language models                                                |

#### Folder Structure

Once you have cloned this repository and finished downloading (or generating) all the additional required files following the instructions in the Jupyter notebooks, your folder structure should be similar to the one below:

- `PHIStruct` (root)
  - `experiments`
    - `data`
      - `GenomesDB` ([Download](https://drive.google.com/file/d/1cOLY_jZOn7gdC_uA2xAN3szIsHj0tI71/view?usp=sharing) and unzip)
        - `AB002632`
        - ...
      - `inphared`
        - `consolidated` ([Download](https://drive.google.com/file/d/1yQSXwlb37dm2ZLXGJHdIM5vmrzwPAwvI/view?usp=sharing) and unzip)
          - `rbp.csv`
          - ...
        - `embeddings`
          - `prottransbert` ([Download](https://drive.google.com/file/d/17B18A9coiX3RUOzr8XzCEZN7TNU5mbQd/view?usp=sharing) and unzip)
            - `complete`
            - `hypothetical`
            - `rbp`
        - `fasta` ([Download](https://drive.google.com/file/d/1NMFR3JrrrCHLoCMQp2nia4dgtcXs5x05/view?usp=sharing) and unzip)
          - `complete`
          - `hypothetical`
          - `nucleotide`
          - `rbp`
        - `structure`
          - `pdb` ([Download](https://drive.google.com/file/d/1ZPRdaHwsFOPksLbOyQerREG0gY0p4-AT/view?usp=sharing) and unzip)
          - `rbp_saprot_embeddings` ([Download](https://drive.google.com/file/d/1l1r41Ze56tXQv_U_KShjECpdaoHffJ8d/view?usp=sharing) and unzip)
            - `AAA74324.1_relaxed.r3.pdb.pt`
          - `rbp_saprot_mask_embeddings` ([Download](https://drive.google.com/file/d/1N6mWO0gG82oP99NqA_pSiAcXFZW6Xk9o/view?usp=sharing) and unzip)
            - `AAA74324.1_relaxed.r3.pdb.pt`
          - `rbp_saprot_seq_mask_embeddings` ([Download](https://drive.google.com/file/d/1__Rok7MoEbTJ3P8iO3Z-bA7_pFUX4CoO/view?usp=sharing) and unzip)
            - `AAA74324.1_relaxed.r3.pdb.pt`
          - `rbp_saprot_struct_mask_embeddings` ([Download](https://drive.google.com/file/d/1GAUsVFQSvKJ2COU1-jUQ5lC3yBUDx9ut/view?usp=sharing) and unzip)
            - `AAA74324.1_relaxed.r3.pdb.pt`
          - `rbp_pst_embeddings` ([Download](https://drive.google.com/file/d/1R18MJSyUGsC7FTr_Fb5qjs8PQwsuEhT9/view?usp=sharing) and unzip)
            - `AAA74324.1_relaxed.r3.pdb.pt`
          - `rbp_prostt5_embeddings.h5` ([Download](https://drive.google.com/file/d/1oNJkzVwTJmy7D38KGOnzn3PsLfDBavmG/view?usp=sharing))
          - `rbp_prostt5_3di_embeddings.h5` ([Download](https://drive.google.com/file/d/1fz56eDOY3q0Ac585gZerQGFQZLsG2y27/view?usp=sharing))
          - `rbp_saprot_mask_relaxed_r3.csv` ([Download](https://drive.google.com/file/d/15M25MbPMmfpk9rAy2I5Y3SlqC4Gi-EId/view?usp=sharing))
          - `rbp_saprot_relaxed_r3.csv` ([Download](https://drive.google.com/file/d/1rY65V6wKvfVzC0AENyERMHJIY0b432r6/view?usp=sharing))
          - `rbp_saprot_seq_mask_relaxed_r3.csv` ([Download](https://drive.google.com/file/d/1TTNlUVcaNaWHXMq4n962JTvFEfvGsbVj/view?usp=sharing))
          - `rbp_saprot_struct_mask_relaxed_r3.csv` ([Download](https://drive.google.com/file/d/1eeQphah4GVjxms8vutlt43HuEFmTUTug/view?usp=sharing))
          - `rbp_pst_relaxed_r3.csv` ([Download](https://drive.google.com/file/d/1VaAtVZOxgSWG2vy53AKw71teE_pjS6ST/view?usp=sharing))
          - `rbp_prostt5_relaxed_r3.csv` ([Download](https://drive.google.com/file/d/1PLrfpkUd37G8jbYInWFoghlw_SGHogSV/view?usp=sharing))
          - `rbp_prostt5_3di_relaxed_r3.csv` ([Download](https://drive.google.com/file/d/1QfUzxwbfK_Lk42SB7aeP7DbTNgJGJy6p/view?usp=sharing))
      - `3Oct2023_data_excluding_refseq.tsv`
      - `3Oct2023_phages_downloaded_from_genbank.gb` ([Download](https://drive.google.com/file/d/1bZbskKri5ecIvPj3KGP0vd_NgVHTdaVT/view?usp=sharing))
    - `preprocessing`
    - `rbp_prediction`
    - `temp`
    - `1. Sequence Preprocessing.ipynb`
    - ...
    - `ClassificationUtil.py`
    - ...

</details>

↑ _Return to [Table of Contents](https://github.com/bioinfodlsu/PHIStruct?tab=readme-ov-file#table-of-contents)._

### Dependencies

**Operating System**: Windows (using [WSL](https://learn.microsoft.com/en-us/windows/wsl/install)), Linux, or macOS

Create a virtual environment with the dependencies installed via Conda (we recommend using [Miniconda](https://docs.anaconda.com/free/miniconda/index.html)):

```
conda env create -f environment_experiments.yaml
```

Activate this environment by running:

```
conda activate PHIStruct-experiments
```

↑ _Return to [Table of Contents](https://github.com/bioinfodlsu/PHIStruct?tab=readme-ov-file#table-of-contents)._

## 💻 Authors

- **Mark Edward M. Gonzales** <br>
  gonzales.markedward@gmail.com

- **Ms. Jennifer C. Ureta** <br>
  jennifer.ureta@gmail.com
- **Dr. Anish M.S. Shrestha** <br>
  anish.shrestha@dlsu.edu.ph

This is a research project under the [Bioinformatics Laboratory](https://bioinfodlsu.com/), [Advanced Research Institute for Informatics, Computing and Networking](https://www.dlsu.edu.ph/research/research-centers/adric/), De La Salle University, Philippines.

This research was partly funded by the [Department of Science and Technology Philippine Council for Health Research and Development](https://www.pchrd.dost.gov.ph/) (DOST-PCHRD) under the [e-Asia JRP 2021 Alternative therapeutics to tackle AMR pathogens (ATTACK-AMR) program](https://www.the-easia.org/jrp/projects/project_76.html).

This research was supported with Cloud TPUs from [Google's TPU Research Cloud (TRC)](https://sites.research.google/trc/about/) and with computing resources from the [Machine Learning eResearch Platform (MLeRP)](https://docs.mlerp.cloud.edu.au/) of Monash University, University of Queensland, and Queensland Cyber Infrastructure Foundation Ltd.

[badge-jupyter]: https://img.shields.io/badge/Jupyter-F37626.svg?&style=flat&logo=Jupyter&logoColor=white
[badge-python]: https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=white
[badge-pandas]: https://img.shields.io/badge/Pandas-2C2D72?style=flat&logo=pandas&logoColor=white
[badge-numpy]: https://img.shields.io/badge/Numpy-777BB4?style=flat&logo=numpy&logoColor=white
[badge-scipy]: https://img.shields.io/badge/SciPy-654FF0?style=flat&logo=SciPy&logoColor=white
[badge-github-actions]: https://img.shields.io/badge/GitHub_Actions-2088FF?style=flat&logo=github-actions&logoColor=white
