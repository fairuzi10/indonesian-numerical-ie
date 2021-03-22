# Numerical Information Extraction from Indonesian Text

This repository contains the source code and the dataset used in my bachelor thesis.

## Installation

1. Clone this repository
2. If you use virtualenv, create new virtual environment
   ```
   virtualenv .env
   ```
3. Install some dependencies needed to run this project
   ```
   pip install -r requirements.txt
   ```

## Training

We use Google Colab to run the jupyter notebook file in `train/` folder.
Run each file from `POS_tag.ipynb`, `NER.ipynb`, and `confidence_value.ipynb`.
Each of them will result binary model file that can be used in `extract/model` for extraction.

## Extraction

We have news, website, and Wikipedia as text source thus we have three python program to extract each of them.
The three python program are located in `extract/` folder.
Run them and the result will be in folder `extract/result/`.

## License

TODO
