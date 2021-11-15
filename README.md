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
Each of them will result binary model file that can be used in `extract/model/` for extraction.
If you want to get our trained binary model rightaway, go to [this Goole Drive folder](https://drive.google.com/drive/folders/1cBZFzSGGbnmfm5L1Usc_rt3xCyqd-DN3?usp=sharing) and download them to the `extract/model/` folder.

## Extraction

We have news, website, and Wikipedia as text source thus we have three python program to extract each of them.
The three python program are located in `extract/` folder.
Run them and the result will be in folder `extract/result/`.

## License

This code is provided under [the licence CC BY-NC-SA 4.0: Attribution-NonCommercial-ShareAlike](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
