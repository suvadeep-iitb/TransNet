# TransNet: Shift Invariant Transformer Network for Power Attack

This repository contains the implementation of TransNet, a shift invariant transformer network for Power Attack ([eprint](https://eprint.iacr.org/2021/827)).

The implementation is composed of the following files:
* **transformer.py:** It contains the code of TransNet model.
* **train_trans.py** It contains the code for training and evaluating the TransNet model.
* **data_utils_\<dataset\>.py:** It contains the code for reading the dataset \<dataset\> where \<dataset\> is one of ASCAD, AES HD, AES RD and DPA contest v4.2.
* **evaluation_utils_\<dataset\>.py:** It contains the code for computing the mean key ranks for the dataset \<dataset\> where \<dataset\> is one of ASCAD, AES HD, AES RD and DPA contest v4.2.
* **exp_script_\<dataset\>.sh:** It is the bash script with proper hyper-parameter setting to perform experiments on dataset \<dataset\>.
* **\<dataset\>\_exp_colab.ipynb:** It is the Google Colab script with proper hyper-parameter setting to perform experiments on dataset \<dataset\>.
* **datasets:** This the folder containing the four datasets: ASCAD, AES HD, AES RD and DPA contest v4.2 in compressed format.

The datasets are obtained from [this repository](https://github.com/gabzai/Methodology-for-efficient-CNN-architectures-in-SCA).

## Data Pre-processing:
* The traces of AES RD dataset have been subtracted by the constant 180 to make the values centered around 0.
* The traces of DPA contest v4.2 have been multiplied by the constant 2000 to extend the range of the values to [-128, 128].

## Citation:
```
@article{DBLP:journals/iacr/HajraSAM21,
  author    = {Suvadeep Hajra and
               Sayandeep Saha and
               Manaar Alam and
               Debdeep Mukhopadhyay},
  title     = {TransNet: Shift Invariant Transformer Network for Power Attack},
  journal   = {{IACR} Cryptol. ePrint Arch.},
  pages     = {827},
  year      = {2021},
  url       = {https://eprint.iacr.org/2021/827},
  timestamp = {Wed, 07 Jul 2021 12:09:31 +0200},
  biburl    = {https://dblp.org/rec/journals/iacr/HajraSAM21.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
