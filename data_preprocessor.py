"""Data preprocessing utilities for NeuroDSL Infinity Studio."""

import pandas as pd

def preprocess_data(input_file, output_file, normalize=False, fill_missing=None):
    """
    Applies selected preprocessing steps to a CSV file.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the processed CSV file.
        normalize (bool): Whether to apply min-max normalization to numeric columns.
        fill_missing (str): Method to fill missing values ('mean', 'median', or a constant).
    """
    df = pd.read_csv(input_file)

    if normalize:
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            min_val = df[col].min()
            max_val = df[col].max()
            df[col] = (df[col] - min_val) / (max_val - min_val)

    if fill_missing:
        if fill_missing == 'mean':
            df.fillna(df.mean(), inplace=True)
        elif fill_missing == 'median':
            df.fillna(df.median(), inplace=True)
        else:
            df.fillna(fill_missing, inplace=True)

    df.to_csv(output_file, index=False)
