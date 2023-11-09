"""
Created By: Muthu Harini Kaliraj
Date: November 2, 2023
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

class preprocessing_class:

    def __init__(self,path):
        self.path = path

    def import_dataset(self, path):
        """
            @Parameters:
                dataframe_train (DataFrame): Original DataFrame
            @does:
                Read the dataset and split to input and actual data
            @Returns:
                Matrix: X_train
                Array: y_train
        """
        dataset = pd.read_csv(path)
        index = self.checking_NaN(dataset)
        dataset = dataset.drop(labels=index, axis=0)
        y = dataset['class']
        X = dataset.drop(dataset.columns[[13]], axis=1)
        return X, y
    
    def checking_NaN(self, dataset):
        """
            @Parameters:
                X_train(DataFrame) : Train dataset
            @does:
                Check for NaN and returns the row index containing NaN
            @Returns:

        """
        nan_index = dataset[dataset.isna().any(axis=1)].index[0]
        return nan_index

    def stats(self, dataset):
        """
            @Parameters:
                dataset_train (DataFrame): Original DataFrame
            @does:
                Gives the Statistics of the dataset
            @Returns:
                Statistics dataframe
        """
        variable_stats = dataset.describe()
        return variable_stats

    def scaling(self, X_train, X_test):
        """
            @Parameters:
                X_train(DataFrame) : Train dataset
                X_test(DataFrame) : Test data
            @does:
                Perform Min Max scaling
            @Returns:
                Scaled
                X_train(DataFrame) : Scaled Train dataset
                X_test(DataFrame) : Scaled Test data
        """
        scaling = MinMaxScaler()
        X_train_columns = X_train.columns
        for i in X_train_columns:
            X_train_array = scaling.fit_transform(X = X_train)
            X_train = pd.DataFrame(X_train_array)
        X_train.columns = X_train_columns
        X_test_columns = X_test.columns
        for i in X_test_columns:
            X_test_array = scaling.fit_transform(X = X_test)
            X_test = pd.DataFrame(X_test_array)
        return X_train, X_test

    def split_dataset(self, X, y):
        """
            @Parameters:
                X(DataFrame): Input Data
                y(Array): Output Data
            @does:
                Splits the dataset to Train and test
            @Returns:
                X_train(DataFrame) : Training Input Data
                X_test(DataFrame) : Testing Input Data
                y_train(Array) : Training output Data
                y_test(Array) : Testing Output Data
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify = y, random_state=0)
        return X_train, X_test, y_train, y_test

    def pcadatset(self, X_t):
        """
            @Parameters:
                X_t(DataFrame): Train dataset
            @does:
                Perform PCa
            @Returns:
                X_train(Dataset): Dataset after removing few columns
        """
        principal = PCA(n_components=3)
        principal.fit(X_t)
        x = principal.transform(X_t)
        return x

    def preprocessing(self):
        """
           @Parameters:

           @does:
               Splits, Clean and Normalizing the dataset
           @Returns:
               normalized_X_train(DataFrame): Preprocessed X_train
               normalized_y_train(Array): Preprocessed y_train
        """
        self.X, self.y = self.import_dataset(self.path)
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_dataset(self.X, self.y)
        dataset_stats = self.stats(self.X_train)
        self.normalized_X_train, self.normalized_X_test = self.scaling(self.X_train, self.X_test)
        return self.normalized_X_train, self.y_train