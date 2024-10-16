# TransNet: Shift Invariant Transformer Network for Side Channel Analysis

This repository contains the implementation of TransNet, a shift invariant transformer network for Side Channel Analysis ([paper](https://link.springer.com/chapter/10.1007/978-3-031-17433-9_16)).

The implementation is composed of the following files:
* **transformer.py:** It contains the code of TransNet model.
* **train_trans.py** It contains the code for training and evaluating the TransNet model.
* **data_utils_\<dataset\>.py:** It contains the code for reading the dataset \<dataset\> where \<dataset\> is one of ASCAD, AES HD, AES RD and DPA contest v4.2.
* **evaluation_utils_\<dataset\>.py:** It contains the code for computing the mean key ranks for the dataset \<dataset\> where \<dataset\> is one of the above.
* **exp_script_\<dataset\>.sh:** It is the bash script with proper hyper-parameter setting to perform experiments on dataset \<dataset\>.
* **\<dataset\>\_exp_colab.ipynb:** It is the Google Colab script with proper hyper-parameter setting to perform experiments on dataset \<dataset\>.
* **datasets:** It is the folder containing the four datasets: ASCAD, AES HD, AES RD and DPA contest v4.2 in compressed format.

The datasets are obtained from [this repository](https://github.com/gabzai/Methodology-for-efficient-CNN-architectures-in-SCA).

## Data Pre-processing:
* The traces of AES RD dataset have been subtracted by the constant 180 to make the values centered around 0.
* The traces of DPA contest v4.2 have been multiplied by the constant 2000 to extend the range of the values to [-128, 128].

## Citation:
```
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
