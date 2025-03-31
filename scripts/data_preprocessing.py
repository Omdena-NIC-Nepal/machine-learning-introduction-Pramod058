import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def read_csv(path):
    return pd.read_csv(path)

def handle_missing_values(df):
    # drops every missing values.
    return df.dropna()

def handling_outliers(df):  
    outlier_info = {}
    for cols in df.columns:
        Q1 = df[cols].quantile(0.25)
        Q3 = df[cols].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[cols] < lower_bound) | (df[cols] > upper_bound)]
        outlier_info[cols] = len(outliers)

        no_outliers = df = df[(df[cols] >= lower_bound) & (df[cols] <= upper_bound)]

    return outlier_info, no_outliers


def normalize_data(x):
    scaler = MinMaxScaler()
    return scaler.fit_transform(x)

def standardize_data(df):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler

def split_data(X, Y, test_size):
    return train_test_split(X, Y, test_size=test_size, random_state=42)