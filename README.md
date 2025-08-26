# TransNet: Shift Invariant Transformer Network for Side Channel Analysis

This repository provides the official implementation of **TransNet**, a shift-invariant Transformer architecture designed for **Side-Channel Analysis (SCA)**. For details, please refer to our [paper (AFRICACRYPT 2022)](https://link.springer.com/chapter/10.1007/978-3-031-17433-9_16).

---

## Repository Structure
- **`transformer.py`** — Implementation of the TransNet model.
- **`train_trans.py`** — Training and evaluation scripts for TransNet.
- **`data_utils_\<dataset\>.py`** — Dataset readers for `<dataset> ∈ {ASCAD, AES HD, AES RD, DPA contest v4.2}`.
- **`evaluation_utils_\<dataset\>.py`** — Utilities to compute mean key ranks for each dataset.
- **`exp_script_\<dataset\>.sh`** — Bash scripts with recommended hyperparameter settings for experiments.
- **`\<dataset\>\_exp_colab.ipynb`** — Google Colab notebooks for quick experiments on each dataset.
- **`datasets/`** — Contains compressed versions of the four datasets (sourced from [this repository](https://github.com/gabzai/Methodology-for-efficient-CNN-architectures-in-SCA)).

---

## Data Pre-processing:
- **AES RD dataset:** traces are shifted by subtracting `180` (centering around `0`).  
- **DPA contest v4.2 dataset:** traces are scaled by multiplying with `2000` (extending range to `[-128, 128]`).  

---

## Tested on
- Python 3.8.10  
- absl-py == 2.3.1 
- numpy == 1.24.3
- scipy == 1.10.1
- h5py == 3.11.0
- tensorflow == 2.13.0

---

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/suvadeep-iitb/TransNet.git
   cd TransNet
   ```
2. **Install dependencies (Python >= 3.8 recommended):**
   ```bash
   pip install -r requirements.txt
   ```
3. Train TransNet:
   ```bash
   bash exp_script_ascad.sh train
   ```
4. Perform Evaluation:
   ```bash
   bash exp_script_ascad.sh test
   ```

You can also open the corresponding Google Colab notebook for an interactive demo.

---

## Citation:
If you use TransNet in your research, please cite:
```bibtex
@inproceedings{DBLP:conf/africacrypt/HajraSAM22,
  author       = {Suvadeep Hajra and
                  Sayandeep Saha and
                  Manaar Alam and
                  Debdeep Mukhopadhyay},
  editor       = {Lejla Batina and
                  Joan Daemen},
  title        = {TransNet: Shift Invariant Transformer Network for Side Channel Analysis},
  booktitle    = {Progress in Cryptology - {AFRICACRYPT} 2022: 13th International Conference
                  on Cryptology in Africa, {AFRICACRYPT} 2022, Fes, Morocco, July 18-20,
                  2022, Proceedings},
  series       = {Lecture Notes in Computer Science},
  volume       = {13503},
  pages        = {371--396},
  publisher    = {Springer Nature Switzerland},
  year         = {2022},
  url          = {https://doi.org/10.1007/978-3-031-17433-9\_16},
  doi          = {10.1007/978-3-031-17433-9\_16},
  timestamp    = {Sun, 10 Dec 2023 00:28:26 +0100},
  biburl       = {https://dblp.org/rec/conf/africacrypt/HajraSAM22.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
