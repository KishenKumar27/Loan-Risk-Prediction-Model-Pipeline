import optuna
import lightgbm as lgb
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import RandomOverSampler
from zenml import step

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def objective(trial, X_train, y_train):
    params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'num_class': len(np.unique(y_train)),  # Ensure this matches y_train_encoded
        'n_jobs': -1,
        'class_weight': 'balanced',
        'device': 'cpu'
    }

    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns
    numerical_cols = X_train.select_dtypes(exclude=["object", "category"]).columns

    numerical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    loglosses = []

    for train_index, val_index in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

        # Apply Random Oversampling
        ros = RandomOverSampler(random_state=42)
        X_tr_resampled, y_tr_resampled = ros.fit_resample(X_tr, y_tr)

        # Ensure consistent labels in train and validation
        all_classes = np.unique(y_train)  # Use full dataset labels

        X_tr_processed = preprocessor.fit_transform(X_tr_resampled)
        X_val_processed = preprocessor.transform(X_val)

        train_data = lgb.Dataset(X_tr_processed, label=y_tr_resampled)
        val_data = lgb.Dataset(X_val_processed, label=y_val)

        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(10)]
        )

        y_pred = model.predict(X_val_processed)

        # Ensure log_loss uses all possible labels
        logloss = log_loss(y_val, y_pred, labels=all_classes)
        loglosses.append(logloss)

    return np.mean(loglosses)

@step(enable_cache=False)
def model_tuning_step(X_train, y_train, n_trials=60):
    le = LabelEncoder()
    y_train_encoded = pd.Series(le.fit_transform(y_train))

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train_encoded), n_trials=n_trials)

    best_params = study.best_params
    best_params['objective'] = 'multiclass'
    best_params['metric'] = 'multi_logloss'
    best_params['num_class'] = len(np.unique(y_train_encoded))
    best_params['n_jobs'] = -1
    
    logging.info({'Best Parameters': best_params})

    
    return best_params
