import pandas as pd
from src.feature_engineering import (
    FeatureEngineer,
    LogTransformation,
    MinMaxScaling,
    OneHotEncoding,
    StandardScaling,
    DateConversion,
    FeatureCreation,
    LabelEncoding
)
from zenml import step


def feature_engineering(
    df: pd.DataFrame, strategy: str = "log", features: list = None
) -> pd.DataFrame:
    """Performs feature engineering using FeatureEngineer and selected strategy."""

    # Ensure features is a list, even if not provided
    if features is None:
        features = []  # or raise an error if features are required

    if strategy == "log":
        engineer = FeatureEngineer(LogTransformation(features))
    elif strategy == "standard_scaling":
        engineer = FeatureEngineer(StandardScaling(features))
    elif strategy == "minmax_scaling":
        engineer = FeatureEngineer(MinMaxScaling(features))
    elif strategy == "onehot_encoding":
        engineer = FeatureEngineer(OneHotEncoding(features))
    elif strategy == "date_conversion":
        engineer = FeatureEngineer(DateConversion(features))
    elif strategy == "feature_creation":
        engineer = FeatureEngineer(FeatureCreation())
    else:
        raise ValueError(f"Unsupported feature engineering strategy: {strategy}")

    transformed_df = engineer.apply_feature_engineering(df)
    return transformed_df

@step(enable_cache=False)
def feature_engineering_step(df: pd.DataFrame) -> pd.DataFrame:
    """Applies date conversion, feature creation, categorical encoding, and numerical scaling."""

    # Convert date columns
    date_columns = ["applicationDate", "originatedDate"]
    df = FeatureEngineer(DateConversion(date_columns)).apply_feature_engineering(df)

    # Feature Creation
    df = FeatureEngineer(FeatureCreation()).apply_feature_engineering(df)
    
    # Encode categorical features
    df = FeatureEngineer(LabelEncoding(columns=['loanStatus'])).apply_feature_engineering(df)
    # categorical_columns = ['payFrequency', 'state', 'leadType', 'fpStatus']
    # df = FeatureEngineer(OneHotEncoding(categorical_columns)).apply_feature_engineering(df)
    
    # Normalize numerical features using StandardScaling strategy
    # numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    # df = FeatureEngineer(StandardScaling(numerical_columns)).apply_feature_engineering(df)
    
    return df



