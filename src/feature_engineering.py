import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler, LabelEncoder
from config import MODEL_DIR, ENCODER_DIR
import pickle

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base Class for Feature Engineering Strategy
# ----------------------------------------------------
# This class defines a common interface for different feature engineering strategies.
# Subclasses must implement the apply_transformation method.
class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to apply feature engineering transformation to the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: A dataframe with the applied transformations.
        """
        pass


# Concrete Strategy for Log Transformation
# ----------------------------------------
# This strategy applies a logarithmic transformation to skewed features to normalize the distribution.
class LogTransformation(FeatureEngineeringStrategy):
    def __init__(self, features):
        """
        Initializes the LogTransformation with the specific features to transform.

        Parameters:
        features (list): The list of features to apply the log transformation to.
        """
        self.features = features

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies a log transformation to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with log-transformed features.
        """
        logging.info(f"Applying log transformation to features: {self.features}")
        df_transformed = df.copy()
        for feature in self.features:
            df_transformed[feature] = np.log1p(
                df[feature]
            )  # log1p handles log(0) by calculating log(1+x)
        logging.info("Log transformation completed.")
        return df_transformed


# Concrete Strategy for Standard Scaling
# --------------------------------------
# This strategy applies standard scaling (z-score normalization) to features, centering them around zero with unit variance.
class StandardScaling(FeatureEngineeringStrategy):
    def __init__(self, features):
        """
        Initializes the StandardScaling with the specific features to scale.

        Parameters:
        features (list): The list of features to apply the standard scaling to.
        """
        self.features = features
        self.scaler = StandardScaler()

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies standard scaling to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with scaled features.
        """
        logging.info(f"Applying standard scaling to features: {self.features}")
        df_transformed = df.copy()
        
        df_transformed[self.features] = self.scaler.fit_transform(df_transformed[self.features])
        logging.info("Standard scaling completed.")
        return df_transformed


# Concrete Strategy for Min-Max Scaling
# -------------------------------------
# This strategy applies Min-Max scaling to features, scaling them to a specified range, typically [0, 1].
class MinMaxScaling(FeatureEngineeringStrategy):
    def __init__(self, features, feature_range=(0, 1)):
        """
        Initializes the MinMaxScaling with the specific features to scale and the target range.

        Parameters:
        features (list): The list of features to apply the Min-Max scaling to.
        feature_range (tuple): The target range for scaling, default is (0, 1).
        """
        self.features = features
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies Min-Max scaling to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with Min-Max scaled features.
        """
        logging.info(
            f"Applying Min-Max scaling to features: {self.features} with range {self.scaler.feature_range}"
        )
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("Min-Max scaling completed.")
        return df_transformed


# Concrete Strategy for One-Hot Encoding
# --------------------------------------
# This strategy applies one-hot encoding to categorical features, converting them into binary vectors.
class OneHotEncoding(FeatureEngineeringStrategy):
    def __init__(self, features):
        """
        Initializes the OneHotEncoding with the specific features to encode.

        Parameters:
        features (list): The list of categorical features to apply the one-hot encoding to.
        """
        self.features = features
        self.encoder = OneHotEncoder(sparse_output=False, drop="first")

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies one-hot encoding to the specified categorical features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with one-hot encoded features.
        """
        logging.info(f"Applying one-hot encoding to features: {self.features}")
        
        # Ensure DataFrame is not empty
        if df.empty:
            raise ValueError("The input DataFrame is completely empty. Nothing to encode.")

        
        df_transformed = df.copy()
        encoded_df = pd.DataFrame(
            self.encoder.fit_transform(df[self.features]),
            columns=self.encoder.get_feature_names_out(self.features),
        )
        df_transformed = df_transformed.drop(columns=self.features).reset_index(drop=True)
        df_transformed = pd.concat([df_transformed, encoded_df], axis=1)
        logging.info("One-hot encoding completed.")
        return df_transformed


# Context Class for Feature Engineering
# -------------------------------------
# This class uses a FeatureEngineeringStrategy to apply transformations to a dataset.
class FeatureEngineer:
    def __init__(self, strategy: FeatureEngineeringStrategy):
        """
        Initializes the FeatureEngineer with a specific feature engineering strategy.

        Parameters:
        strategy (FeatureEngineeringStrategy): The strategy to be used for feature engineering.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: FeatureEngineeringStrategy):
        """
        Sets a new strategy for the FeatureEngineer.

        Parameters:
        strategy (FeatureEngineeringStrategy): The new strategy to be used for feature engineering.
        """
        logging.info("Switching feature engineering strategy.")
        self._strategy = strategy

    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the feature engineering transformation using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with applied feature engineering transformations.
        """
        logging.info("Applying feature engineering strategy.")
        return self._strategy.apply_transformation(df)
    
    
# Concrete Strategy for Date Conversion and Feature Extraction
class DateConversion(FeatureEngineeringStrategy):
    def __init__(self, date_columns):
        self.date_columns = date_columns

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Converting columns to datetime and extracting features: {self.date_columns}")
        df_transformed = df.copy()
        for col in self.date_columns:
            if col in df_transformed.columns:
                df_transformed[col] = pd.to_datetime(df_transformed[col], errors='coerce')
                df_transformed[f"{col}_year"] = df_transformed[col].dt.year
                df_transformed[f"{col}_month"] = df_transformed[col].dt.month
                df_transformed[f"{col}_day"] = df_transformed[col].dt.day
        df_transformed = df_transformed.drop(columns=self.date_columns)
        logging.info("Date conversion and feature extraction completed.")
        return df_transformed
    

class FeatureCreation(FeatureEngineeringStrategy):
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Creating new features: is_monthly_payment and loan_to_payment_ratio")
        df_transformed = df.copy()
        
        if 'payFrequency' in df_transformed.columns:
            df_transformed['is_monthly_payment'] = df_transformed['payFrequency'].apply(lambda x: 1 if x == 'M' else 0)
        
        if 'loanAmount' in df_transformed.columns and 'originallyScheduledPaymentAmount' in df_transformed.columns:
            df_transformed['loan_to_payment_ratio'] = df_transformed['loanAmount'] / df_transformed['originallyScheduledPaymentAmount']
            
        df_transformed.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        logging.info("Feature creation completed.")
        return df_transformed
    
# Concrete Strategy for Label Encoding
# -------------------------------------
# This strategy applies label encoding to all categorical features in the DataFrame.
class LabelEncoding(FeatureEngineeringStrategy):
    def __init__(self, columns=None):
        """
        Initializes the LabelEncoding with specified columns for encoding.
        
        Parameters:
        columns (list, optional): List of column names to apply label encoding. If None, an error is raised.
        """
        self.encoder = LabelEncoder()
        self.columns = columns  # Store user-specified categorical columns
    
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies label encoding to specified categorical features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with label-encoded categorical features.
        """
        if df.empty:
            raise ValueError("The input DataFrame is completely empty. Nothing to encode.")
        
        if not self.columns:
            raise ValueError("No columns specified for label encoding. Please provide a list of columns.")
        
        df_transformed = df.copy()
        
        for feature in self.columns:
            if feature not in df_transformed.columns:
                raise ValueError(f"Column '{feature}' not found in DataFrame.")
            
            if df_transformed[feature].dtype not in ['object', 'category']:
                raise ValueError(f"Column '{feature}' is not categorical and cannot be label encoded.")
            
            logging.info(f"Applying label encoding to column: {feature}")
            df_transformed[feature] = self.encoder.fit_transform(df_transformed[feature])
            
        # Generate encoder filename based on timestamp or versioning logic
        model_filename = f"{ENCODER_DIR}/label_encoder_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        # Save the model
        with open(model_filename, "wb") as f:
            pickle.dump(self.encoder, f)
        logging.info(f"Label Encoder saved to {model_filename}")
        
        logging.info("Label encoding completed.")
        
        return df_transformed


# Example usage
if __name__ == "__main__":
    # Example dataframe
    # df = pd.read_csv('../extracted-data/your_data_file.csv')

    # Log Transformation Example
    # log_transformer = FeatureEngineer(LogTransformation(features=['SalePrice', 'Gr Liv Area']))
    # df_log_transformed = log_transformer.apply_feature_engineering(df)

    # Standard Scaling Example
    # standard_scaler = FeatureEngineer(StandardScaling(features=['SalePrice', 'Gr Liv Area']))
    # df_standard_scaled = standard_scaler.apply_feature_engineering(df)

    # Min-Max Scaling Example
    # minmax_scaler = FeatureEngineer(MinMaxScaling(features=['SalePrice', 'Gr Liv Area'], feature_range=(0, 1)))
    # df_minmax_scaled = minmax_scaler.apply_feature_engineering(df)

    # One-Hot Encoding Example
    # onehot_encoder = FeatureEngineer(OneHotEncoding(features=['Neighborhood']))
    # df_onehot_encoded = onehot_encoder.apply_feature_engineering(df)

    pass
