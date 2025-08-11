# Standard libraries
import warnings
from itertools import product

# Data manipulation
import numpy as np
import pandas as pd

# Machine learning models
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Preprocessing and model evaluation
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_recall_curve, confusion_matrix, auc,
    balanced_accuracy_score, matthews_corrcoef
)

from pre_processing import *

# Suppress warnings
warnings.filterwarnings('ignore')


def prepare_features_and_targets(df, test_set=0, target_var='responsive'):
    if target_var not in df.columns:
        raise ValueError(f"Target variable '{target_var}' not found in dataframe.")

    # Extract target variable and drop excluded columns
    targets = df[target_var]
    features = df[[col for col in df.columns if col != target_var and col != 'id']]
    features = features.drop(columns=[target_var], errors='ignore')

    # Split into training and test sets (STRATIFIED)
    if test_set:
        X_train, X_test, Y_train, Y_test = train_test_split(
            features, targets, test_size=test_set, stratify=targets
        )
    else: 
        X_train = features
        Y_train = targets
        X_test = []
        Y_test = []

    # Median imputation for 'income_numeric' if it contains NA values
    if 'income_numeric' in X_train.columns:
        if X_train['income_numeric'].isna().any():
            X_train['income_numeric'].fillna(X_train['income_numeric'].median(), inplace=True)
        if isinstance(X_test, pd.DataFrame) and 'income_numeric' in X_test.columns and X_test['income_numeric'].isna().any():
            X_test['income_numeric'].fillna(X_test['income_numeric'].median(), inplace=True)

    if 'IAS_mean' in X_train.columns:
        if X_train['IAS_mean'].isna().any():
            X_train['IAS_mean'].fillna(X_train['IAS_mean'].median(), inplace=True)
        if isinstance(X_test, pd.DataFrame) and 'IAS_mean' in X_test.columns and X_test['IAS_mean'].isna().any():
            X_test['IAS_mean'].fillna(X_test['IAS_mean'].median(), inplace=True)

    return X_train, Y_train, X_test, Y_test


def random_forest_kfold_grid_search(
    X, Y, param_grid, k=5, CV_reps=1, eval_metric=['auc'], model_choice_metric='auc', 
    res_dir=".", model_type='rf', combo='alcohol'
):

    # Generate all parameter combinations
    param_combinations = list(product(*param_grid.values()))
    param_names = list(param_grid.keys())

    # Initialize variables to store the best model and scores
    best_model = None
    best_scores = None
    best_params = None
    best_model_choice_value = -np.inf  # Track the best model based on the chosen metric

    kf = StratifiedKFold(n_splits=k, shuffle=True)

    for params in param_combinations:
        current_params = dict(zip(param_names, params))

        # Store all fold results
        all_folds_metrics = {metric: [] for metric in eval_metric}

        for train_index, test_index in kf.split(X, Y):  # k-fold cv split

            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

            rep_metrics = {metric: [] for metric in eval_metric}  # Reset for each fold

            for _ in range(CV_reps):  # Repeat that split j times

                # Initialize the model with the current parameters
                if model_type == 'rf':
                    model = RandomForestClassifier(
                        n_estimators=current_params.get("n_estimators", 100),
                        max_depth=current_params.get("max_depth"),
                        min_samples_split=current_params.get("min_samples_split", 2),
                        min_samples_leaf=current_params.get("min_samples_leaf", 1),
                        class_weight="balanced"
                    )
                elif model_type == 'xgb':
                    model = XGBClassifier(
                        n_estimators=current_params.get("n_estimators", 100),
                        max_depth=current_params.get("max_depth", 6),
                        learning_rate=current_params.get("learning_rate", 0.1),
                        min_child_weight=current_params.get("min_child_weight", 1),
                        gamma=current_params.get("gamma", 0),
                        subsample=current_params.get("subsample", 1),
                        colsample_bytree=current_params.get("colsample_bytree", 1),
                        scale_pos_weight=current_params.get("scale_pos_weight", 1),
                        use_label_encoder=False,
                        eval_metric="logloss"
                    )

                model.fit(X_train, Y_train)
                Y_pred = model.predict(X_test)
                Y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

                if 'auc' in eval_metric and Y_prob is not None:
                    rep_metrics['auc'].append(roc_auc_score(Y_test, Y_prob))
                if 'f1' in eval_metric:
                    rep_metrics['f1'].append(f1_score(Y_test, Y_pred))
                if 'accuracy' in eval_metric:
                    rep_metrics['accuracy'].append(accuracy_score(Y_test, Y_pred))
                if 'specificity' in eval_metric or 'sensitivity' in eval_metric:
                    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
                    if 'specificity' in eval_metric:
                        rep_metrics['specificity'].append(specificity)
                    if 'sensitivity' in eval_metric:
                        rep_metrics['sensitivity'].append(sensitivity)
                if 'mcc' in eval_metric:
                    rep_metrics['mcc'].append(matthews_corrcoef(Y_test, Y_pred))
                if 'balancedAcc' in eval_metric:
                    rep_metrics['balancedAcc'].append(balanced_accuracy_score(Y_test, Y_pred))
                if 'pr_auc' in eval_metric and Y_prob is not None:
                    precision, recall, _ = precision_recall_curve(Y_test, Y_prob)
                    pr_auc = auc(recall, precision)
                    rep_metrics['pr_auc'].append(pr_auc)

            # Compute median scores per fold and store results
            fold_median_metrics = {metric: np.mean(values) for metric, values in rep_metrics.items()}
            for metric in eval_metric:
                all_folds_metrics[metric].append(fold_median_metrics[metric])

        # Compute final median scores over all folds
        median_rep_metrics = {metric: np.mean(values) for metric, values in all_folds_metrics.items()}

        # Select best model based on median of model_choice_metric
        if median_rep_metrics[model_choice_metric] > best_model_choice_value:
            best_model_choice_value = median_rep_metrics[model_choice_metric]
            best_model = model
            best_params = current_params
            best_scores = median_rep_metrics  # Store median scores for all metrics

    return best_model, best_scores, best_params
