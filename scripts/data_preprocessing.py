import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def read_csv(path):
    return pd.read_csv(path)

def handle_missing_values(df):
    # drops every missing values.
    df = df.dropna()
    return df

def handling_outliers(df, column):
    
    """Detects outliers in a given column using the IQR method."""

    if column not in df.columns:
        return f"Column '{column}' not found in DataFrame."
    
    # Compute Q1, Q3, and IQR
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define outlier range
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter outliers
    outliers = df[column][(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    return outliers


def encode_categorical_vars(df, column_name):
    '''This returns the changed data of catergorical values by separating based on the status
    so that ML runs effectively'''

    dummies = pd.get_dummies(df[column_name])
    # change True or False to yes or no. So it is same as other data.
    dummies = dummies.map(lambda x: 'yes' if x == 1 else 'no')

    return dummies

def encode_bools_to_binary(df, bool_columns):
    df[bool_columns] = df[bool_columns].apply(lambda x: x.map({'yes': 1, 'no': 0}))
    
    return df


def standardize_data(df):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler

def split_data(X, Y, test_size):
    return train_test_split(X, Y, test_size=test_size, random_state=42)