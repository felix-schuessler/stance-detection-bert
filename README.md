# Stance Detection with BERT

## Description / Datasets

To run the Stance Detection Model 2 datasets are required. train.csv and predict.csv.
In the file bert_stance.py change the TRAIN_CSV_CLEAN and PREDICT_CSV_CLEAN variable to point to your .csv files

- train.csv (must contain tweet and label column)
  - The label colum must include the categories (AGAINST, FAVOR, NONE)
- predict.csv (must contains tweets column)

## Installing

```sh

# create conda env
conda create --name stance python=3.10.8
conda activate stance
python --version

# 1. install dependencies from .yaml (mac)
conda env update --file env.yml --prune

# 2. install dependencies from .yaml (window)
conda env update --file env_win.yml --prune

# 3. install dependencies manually
conda install pandas==2.2.2 numpy==1.26.4 scikit-learn==1.5.1
pip install tensorflow==2.10 transformers==4.43.2


# remove the env (optional)
conda deactivate && conda remove --name stance --all
```
