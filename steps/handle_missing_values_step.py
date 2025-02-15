import pandas as pd
from src.handle_missing_values import (
    DropMissingValuesStrategy,
    FillMissingValuesStrategy,
    MissingValueHandler,
)
from zenml import step
import logging
from typing import Optional, List


def handle_all_missing_values(df: pd.DataFrame, columns: Optional[List[str]] = None, strategy: str = "median", column_type='numeric') -> pd.DataFrame:
    """Handles missing values using MissingValueHandler and the specified strategy.

        Args:
            df (pd.DataFrame): Input DataFrame.
            columns (list, optional): Specific columns to apply the strategy on. Defaults to None.
            strategy (str, optional): Strategy to handle missing values. Defaults to "median".
            column_type (str, optional): Type of columns to apply ("numeric" or "categorical"). Defaults to "numeric".

        Returns:
            pd.DataFrame: Cleaned DataFrame.
    """

    # Handle missing values based on strategy
    if strategy == "drop":
        handler = MissingValueHandler(DropMissingValuesStrategy(axis=1, columns=columns))
    elif strategy in ["mean", "median", "mode", "constant"]:
        handler = MissingValueHandler(FillMissingValuesStrategy(method=strategy, column_type=column_type))
    else:
        raise ValueError(f"Unsupported missing value handling strategy: {strategy}")

    cleaned_df = handler.handle_missing_values(df)
    return cleaned_df

@step
def handle_missing_values_step(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values in one step."""
    
    # Drop irrelevant columns
    irrelevant_cols = ["loanId", "anon_ssn", "clarityFraudId"]
    df = handle_all_missing_values(df=df, columns=irrelevant_cols, strategy="drop")
    
    # Fill numeric missing values with mean
    df = handle_all_missing_values(df=df, strategy="mean", column_type="numeric")

    # Fill categorical missing values with mode
    df = handle_all_missing_values(df=df, strategy="mode", column_type="object")

    return df