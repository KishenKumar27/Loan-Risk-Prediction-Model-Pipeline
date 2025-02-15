import logging
from abc import ABC, abstractmethod

import pandas as pd

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Abstract Base Class for Missing Value Handling Strategy
class MissingValueHandlingStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to handle missing values in the DataFrame.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing missing values.

        Returns:
        pd.DataFrame: The DataFrame with missing values handled.
        """
        pass


# Concrete Strategy for Dropping Missing Values
class DropMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, axis=0, thresh=None, columns=None):
        """
        Initializes the DropMissingValuesStrategy with specific parameters.

        Parameters:
        axis (int): 0 to drop rows with missing values, 1 to drop columns with missing values.
        thresh (int): The threshold for non-NA values. Rows/Columns with less than thresh non-NA values are dropped.
        """
        self.axis = axis
        self.thresh = thresh
        self.columns = columns

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops rows or columns with missing values based on the axis and threshold.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing missing values.

        Returns:
        pd.DataFrame: The DataFrame with missing values dropped.
        """
        logging.info(f"Dropping missing values with axis={self.axis}, thresh={self.thresh} and columns={self.columns}")
        
        if self.axis == 1:
            # Dropping columns directly (subset is not required)
            df_cleaned = df.drop(columns=self.columns, axis=1)
        else:
            # Dropping rows where these columns have missing values
            df_cleaned = df.dropna(axis=0, subset=self.columns, thresh=self.thresh)        
            
        logging.info("Missing values dropped.")
        return df_cleaned


# Concrete Strategy for Filling Missing Values
class FillMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, method="mean", fill_value='', column_type=None):
        """
        Initializes the FillMissingValuesStrategy with a specific method or fill value.

        Parameters:
        method (str): The method to fill missing values ('mean', 'median', 'mode', or 'constant').
        fill_value (any): The constant value to fill missing values when method='constant'.
        """
        self.method = method
        self.fill_value = fill_value
        self.column_type = column_type

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values using the specified method or constant value.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing missing values.

        Returns:
        pd.DataFrame: The DataFrame with missing values filled.
        """
        logging.info(f"Filling missing values using method: {self.method}")

        df_cleaned = df.copy()
        numeric_columns = df_cleaned.select_dtypes(include="number").columns
        object_columns = df_cleaned.select_dtypes(include="object").columns

        
        if self.method == "mean":
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                df[numeric_columns].mean()
            )
        elif self.method == "median":
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                df[numeric_columns].median()
            )
        elif self.method == "mode":
            columns_to_process = []
            if self.column_type == "numeric":
                columns_to_process = numeric_columns
            elif self.column_type == "object":
                columns_to_process = object_columns
            elif self.column_type == "both":
                columns_to_process = list(numeric_columns) + list(object_columns)
            else:
                logging.warning(f"Invalid column_type '{self.column_type}'. Must be 'numeric', 'object', or 'both'.")
                return df_cleaned

            for column in columns_to_process:
                if df[column].dropna().empty:
                    logging.warning(f"Column '{column}' is entirely missing. Filling with provided fill_value.")
                    df_cleaned[column] = df_cleaned[column].fillna(self.fill_value)
                    continue

                mode_value = df[column].mode()
                if not mode_value.empty:
                    df_cleaned[column] = df_cleaned[column].fillna(mode_value.iloc[0])
                else:
                    logging.warning(f"No mode found for column '{column}'. Using fill_value instead.")
                    df_cleaned[column] = df_cleaned[column].fillna(self.fill_value)
                    
        elif self.method == "constant":
            df_cleaned = df_cleaned.fillna(self.fill_value)
            
        else:
            logging.warning(f"Unknown method '{self.method}'. No missing values handled.")

        # Log remaining missing values if any
        missing_counts = df_cleaned.isnull().sum()
        if missing_counts.any():
            logging.warning("Some columns still have missing values:")
            for col, count in missing_counts[missing_counts > 0].items():
                logging.warning(f"Column '{col}': {count} missing values")
        else:
            logging.info("All missing values successfully filled.")

        return df_cleaned


# Context Class for Handling Missing Values
class MissingValueHandler:
    def __init__(self, strategy: MissingValueHandlingStrategy):
        """
        Initializes the MissingValueHandler with a specific missing value handling strategy.

        Parameters:
        strategy (MissingValueHandlingStrategy): The strategy to be used for handling missing values.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: MissingValueHandlingStrategy):
        """
        Sets a new strategy for the MissingValueHandler.

        Parameters:
        strategy (MissingValueHandlingStrategy): The new strategy to be used for handling missing values.
        """
        logging.info("Switching missing value handling strategy.")
        self._strategy = strategy

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the missing value handling using the current strategy.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing missing values.

        Returns:
        pd.DataFrame: The DataFrame with missing values handled.
        """
        logging.info("Executing missing value handling strategy.")
        return self._strategy.handle(df)


# Example usage
if __name__ == "__main__":
    # Example dataframe
    # df = pd.read_csv('../extracted-data/your_data_file.csv')

    # Initialize missing value handler with a specific strategy
    # missing_value_handler = MissingValueHandler(DropMissingValuesStrategy(axis=0, thresh=3))
    # df_cleaned = missing_value_handler.handle_missing_values(df)

    # Switch to filling missing values with mean
    # missing_value_handler.set_strategy(FillMissingValuesStrategy(method='mean'))
    # df_filled = missing_value_handler.handle_missing_values(df)

    pass
