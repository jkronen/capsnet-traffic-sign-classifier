#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import pickle
from sklearn.model_selection import train_test_split


TRAIN_FILE = "train48.p"
#VALID_FILE = "valid.p"
TEST_FILE = "test48.p"

def get_data(folder):
    """
        Load traffic sign data
        **input: **
            *folder: (String) Path to the dataset folder
    """
    # Load the dataset
    training_file = os.path.join(folder, TRAIN_FILE)
    #validation_file= os.path.join(folder, VALID_FILE)
    testing_file =  os.path.join(folder, TEST_FILE)

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    #with open(validation_file, mode='rb') as f:
    #    valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    # Retrive all datas
    X_train, y_train = train['features'], train['labels']
    #X_valid, y_valid = valid['features'], valid['labels']
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    X_test, y_test = test['features'], test['labels']

    return X_train, y_train, X_valid, y_valid, X_test, y_test
