"""
split_data.py: Splitting Data into Train/Dev/Test Sets
"""

import numpy as np
import pandas as pd
import os


def split_data(file="http://wwbp.org/downloads/public_data/dataset-fb-valence-arousal-anon.csv"):
    """
    download data, split it into train/dev/test sets, and save files
    
    @param file (string): location of datafile to download
    """
    TRAIN_FILE = "data/train.csv"
    DEV_FILE = "data/dev.csv"
    TEST_FILE = "data/test.csv"
    np.random.seed(0)
    
    
    if not (os.path.isfile(TRAIN_FILE) and os.path.isfile(DEV_FILE) and os.path.isfile(TEST_FILE)):
        df = pd.read_csv(file).dropna()
        train_msk = np.random.rand(len(df)) < 0.65
        train_dev = df[train_msk]
        test = df[~train_msk]
        
        dev_msk = np.random.randn(len(train_dev)) < 0.75
        train = train_dev[dev_msk]
        dev = train_dev[~dev_msk]
        
        print("test: "+str(len(test)), "train: "+str(len(train)), "dev: "+str(len(dev)))
        train.to_csv (TRAIN_FILE, index = None, header=True)
        dev.to_csv (DEV_FILE, index = None, header=True)
        test.to_csv (TEST_FILE, index = None, header=True)
        
    
    
if __name__ == '__main__':
    split_data()