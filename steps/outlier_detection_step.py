import logging

import pandas as pd
from src.outlier_detection import OutlierDetector, ZScoreOutlierDetection, IQROutlierDetection
from zenml import step
from typing import List


@step
def outlier_detection_step(df: pd.DataFrame, column_names: List[str]) -> pd.DataFrame:
    """Detects and removes outliers using OutlierDetector for specified columns."""
    logging.info(f"Starting outlier detection step with DataFrame of shape: {df.shape}")

    if df is None:
        logging.error("Received a NoneType DataFrame.")
        raise ValueError("Input df must be a non-null pandas DataFrame.")

    if not isinstance(df, pd.DataFrame):
        logging.error(f"Expected pandas DataFrame, got {type(df)} instead.")
        raise ValueError("Input df must be a pandas DataFrame.")

    if not isinstance(column_names, list) or not all(isinstance(col, str) for col in column_names):
        logging.error("Column names should be provided as a list of strings.")
        raise ValueError("Column names should be a list of strings.")

    missing_columns = [col for col in column_names if col not in df.columns]
    if missing_columns:
        logging.error(f"Columns {missing_columns} do not exist in the DataFrame.")
        raise ValueError(f"Columns {missing_columns} do not exist in the DataFrame.")

    # Ensure only numeric columns are selected for outlier removal
    df_numeric = df[column_names].select_dtypes(include=[int, float])
    if df_numeric.empty:
        logging.error("No numeric columns found among the specified columns.")
        raise ValueError("No numeric columns found among the specified columns.")

    # Perform outlier detection and removal
    outlier_detector = OutlierDetector(IQROutlierDetection())
    df_cleaned_numeric = outlier_detector.handle_outliers(df_numeric, method="remove")

    # Align indices to match the original DataFrame after outlier removal
    df_cleaned = df.loc[df_cleaned_numeric.index].copy()
    df_cleaned[column_names] = df_cleaned_numeric

    return df_cleaned

