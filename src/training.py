import warnings
from itertools import product

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_recall_curve,
    confusion_matrix,
    auc,
    balanced_accuracy_score,
    matthews_corrcoef,
)

from pygam import LogisticGAM, s, f

from pre_processing import *

warnings.filterwarnings("ignore")


def prepare_features_and_targets(df, test_set=0, target_var='responsive'):
    if target_var not in df.columns:
        raise ValueError(f"Target variable '{target_var}' not found in dataframe.")

    # Extract target variable and drop excluded columns
    targets = df[target_var]
    features = df[[col for col in df.columns if col != target_var and col != 'id']]
    features = features.drop(columns=[target_var], errors='ignore')

    ids = df['id'] if 'id' in df.columns else pd.Series(np.arange(len(df)))

    # # Split into training and test sets (STRATIFIED)
    # if test_set:
    #     X_train, X_test, Y_train, Y_test = train_test_split(
    #         features, targets, test_size=test_set, stratify=targets
    #     )
    # else: 
    #     X_train = features
    #     Y_train = targets
    #     X_test = []
    #     Y_test = []

    if test_set:
        X_train, X_test, Y_train, Y_test, id_train, id_test = train_test_split(
            features, targets, ids, test_size=test_set, stratify=targets
        )
    else: 
        X_train = features
        Y_train = targets
        X_test = pd.DataFrame()
        Y_test = pd.Series(dtype=targets.dtype)
        id_test = pd.Series(dtype=ids.dtype)

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

    return X_train, Y_train, X_test, Y_test, id_test


def random_forest_kfold_grid_search(
    X, Y, param_grid, k=5, CV_reps=1, eval_metric=['auc'], model_choice_metric='auc', 
    res_dir=".", model_type='rf', combo='alcohol', random_state=321
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
                elif model_type == 'dt':
                    model = DecisionTreeClassifier(
                    max_depth=current_params.get("max_depth", None),
                    min_samples_split=current_params.get("min_samples_split", 2),
                    min_samples_leaf=current_params.get("min_samples_leaf", 1),
                    class_weight="balanced",
                    random_state=random_state
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


from itertools import product
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, confusion_matrix,
    matthews_corrcoef, balanced_accuracy_score, precision_recall_curve, auc
)

def logistic_regression_kfold_grid_search(
    X, Y, param_grid, k=5, CV_reps=1, eval_metric=['auc'], model_choice_metric='auc', 
    res_dir=".", combo='alcohol'
):

    param_combinations = []
    param_names = list(param_grid.keys())

    for combo in product(*param_grid.values()):
        params = dict(zip(param_names, combo))
        solver = params.get("solver")
        penalty = params.get("penalty")

        # --- Type sanity check ---
        if not isinstance(params.get("C"), (int, float)):
            continue

        # --- Compatibility rules ---
        if solver == "liblinear" and penalty not in ["l1", "l2"]:
            continue  # liblinear only supports l1 or l2
        if solver == "lbfgs" and penalty not in ["l2", None]:
            continue  # lbfgs supports only l2 or None
        if solver == "newton-cg" and penalty not in ["l2", None]:
            continue
        if solver == "saga" and penalty is None:
            continue  # saga supports l1, l2, elasticnet but not None
        if solver != "saga" and penalty == "elasticnet":
            continue  # only saga supports elasticnet
        if penalty == "elasticnet" and solver == "saga":
            # if elasticnet, add l1_ratio to avoid runtime error
            for l1_ratio in [0.0, 0.5, 1.0]:
                params_with_ratio = params.copy()
                params_with_ratio["l1_ratio"] = l1_ratio
                param_combinations.append(params_with_ratio)
            continue  # skip adding base params (handled above)

        # --- If all checks passed ---
        param_combinations.append(params)

    # Initialize variables to store the best model and scores
    best_model = None
    best_scores = None
    best_params = None
    best_model_choice_value = -np.inf  # Track the best model based on the chosen metric

    kf = StratifiedKFold(n_splits=k, shuffle=True)

    for params in param_combinations:
        current_params = params.copy()

        # Store all fold results
        all_folds_metrics = {metric: [] for metric in eval_metric}

        for train_index, test_index in kf.split(X, Y):  # k-fold cv split

            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

            rep_metrics = {metric: [] for metric in eval_metric}  # Reset for each fold

            for _ in range(CV_reps):  # Repeat that split j times

                # Initialize the logistic regression model with current parameters
                if current_params.get("penalty") == "elasticnet":
                    if l1_ratio is None:
                        # skip invalid combo
                        continue
                    model = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(
                            penalty=current_params.get("penalty", "elasticnet"),
                            C=current_params.get("C", 1.0),
                            solver=current_params.get("solver", "saga"),  # saga required for elasticnet
                            max_iter=current_params.get("max_iter", 1000),
                            class_weight="balanced",
                            l1_ratio=current_params.get("l1_ratio", 1.0)
                        )
                    )
                else:
                    model = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(
                            penalty=current_params.get("penalty", "l2"),
                            C=current_params.get("C", 1.0),
                            solver=current_params.get("solver", "lbfgs"),
                            max_iter=current_params.get("max_iter", 1000),
                            class_weight="balanced"
                        )
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

            # Compute mean scores per fold
            fold_median_metrics = {metric: np.mean(values) for metric, values in rep_metrics.items()}
            for metric in eval_metric:
                all_folds_metrics[metric].append(fold_median_metrics[metric])

        # Compute final mean scores over all folds
        median_rep_metrics = {metric: np.mean(values) for metric, values in all_folds_metrics.items()}

        # Select best model based on chosen metric
        if median_rep_metrics[model_choice_metric] > best_model_choice_value:
            best_model_choice_value = median_rep_metrics[model_choice_metric]
            best_model = model
            best_params = current_params
            best_scores = median_rep_metrics

    return best_model, best_scores, best_params

from itertools import product
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, confusion_matrix,
    matthews_corrcoef, balanced_accuracy_score, precision_recall_curve, auc
)

def svm_kfold_grid_search(
    X, Y, param_grid, k=5, CV_reps=1, eval_metric=['auc'],
    model_choice_metric='auc', res_dir=".", combo='alcohol'
):
    # Build explicit param combinations (mirroring your LR function)
    param_combinations = []
    param_names = list(param_grid.keys())

    for combo_vals in product(*param_grid.values()):
        params = dict(zip(param_names, combo_vals))

        # --- Type sanity check ---
        if not isinstance(params.get("C"), (int, float)):
            continue

        kernel = params.get("kernel", "rbf")
        gamma = params.get("gamma", "scale")
        degree = params.get("degree", 3)

        # --- Compatibility rules ---
        # gamma is ignored for linear kernel; fine to keep
        if kernel not in ["linear", "rbf", "poly", "sigmoid"]:
            continue
        if kernel == "poly" and not isinstance(degree, (int, np.integer)):
            continue

        # If all checks passed
        param_combinations.append(params)

    best_model = None
    best_scores = None
    best_params = None
    best_model_choice_value = -np.inf

    kf = StratifiedKFold(n_splits=k, shuffle=True)

    for params in param_combinations:
        current_params = params.copy()

        # Store all fold results
        all_folds_metrics = {metric: [] for metric in eval_metric}

        for train_index, test_index in kf.split(X, Y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

            rep_metrics = {metric: [] for metric in eval_metric}

            for _ in range(CV_reps):
                # Initialize SVM with current params
                # model = SVC(
                #     kernel=current_params.get("kernel", "rbf"),
                #     C=current_params.get("C", 1.0),
                #     gamma=current_params.get("gamma", "scale"),
                #     degree=current_params.get("degree", 3),
                #     probability=True,  # to enable predict_proba for AUC/PR-AUC
                #     class_weight="balanced"
                # )

                model = make_pipeline(
                    StandardScaler(),
                    SVC(
                        C=current_params.get("C", 1.0),
                        kernel=current_params.get("kernel", "rbf"),
                        gamma=current_params.get("gamma", "scale"),
                        degree=current_params.get("degree", 3),
                        probability=True,
                        class_weight="balanced"
                    )
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
                if 'PPV' in eval_metric or 'NPV' in eval_metric:
                    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
                    PPV = tp / (tp + fp) if (tp + fp) > 0 else np.nan
                    NPV = tn / (tn + fn) if (tn + fn) > 0 else np.nan
                    if 'PPV' in eval_metric:
                        rep_metrics['PPV'].append(PPV)
                    if 'NPV' in eval_metric:
                        rep_metrics['NPV'].append(NPV)
                if 'MCC' in eval_metric or 'mcc' in eval_metric:
                    mcc_val = matthews_corrcoef(Y_test, Y_pred)
                    if 'MCC' in eval_metric:
                        rep_metrics['MCC'].append(mcc_val)
                    if 'mcc' in eval_metric:
                        rep_metrics['mcc'].append(mcc_val)
                if 'balancedAcc' in eval_metric:
                    rep_metrics['balancedAcc'].append(balanced_accuracy_score(Y_test, Y_pred))
                if 'pr_auc' in eval_metric and Y_prob is not None:
                    precision, recall, _ = precision_recall_curve(Y_test, Y_prob)
                    pr_auc = auc(recall, precision)
                    rep_metrics['pr_auc'].append(pr_auc)
                if 'tn' in eval_metric or 'fn' in eval_metric or 'tp' in eval_metric or 'fp' in eval_metric:
                    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
                    if 'tn' in eval_metric: rep_metrics.setdefault('tn', []).append(tn)
                    if 'fp' in eval_metric: rep_metrics.setdefault('fp', []).append(fp)
                    if 'fn' in eval_metric: rep_metrics.setdefault('fn', []).append(fn)
                    if 'tp' in eval_metric: rep_metrics.setdefault('tp', []).append(tp)

            # Mean over CV_reps for this fold
            fold_mean_metrics = {metric: np.mean(values) for metric, values in rep_metrics.items()}
            for metric in eval_metric:
                all_folds_metrics[metric].append(fold_mean_metrics.get(metric, np.nan))

        # Mean over folds
        mean_over_folds = {metric: np.mean(values) for metric, values in all_folds_metrics.items()}

        # Select best model by chosen metric
        if mean_over_folds.get(model_choice_metric, -np.inf) > best_model_choice_value:
            best_model_choice_value = mean_over_folds[model_choice_metric]
            best_model = model
            best_params = current_params
            best_scores = mean_over_folds

    return best_model, best_scores, best_params

def gam_kfold_grid_search(
    X, Y, param_grid, k=5, CV_reps=1, eval_metric=['auc'],
    model_choice_metric='auc', res_dir=".", combo='alcohol'
):
    """
    Cross-validated grid search for LogisticGAM with conservative overfitting control.
    Mirrors the SVM version but fits penalized smooth models.
    """

    # --- Build param combinations ---
    param_combinations = []
    param_names = list(param_grid.keys())
    for combo_vals in product(*param_grid.values()):
        params = dict(zip(param_names, combo_vals))
        param_combinations.append(params)

    best_model = None
    best_scores = None
    best_params = None
    best_model_choice_value = -np.inf

    kf = StratifiedKFold(n_splits=k, shuffle=True)

    # --- Scale continuous predictors ---
    X_scaled = pd.DataFrame(
        StandardScaler().fit_transform(X),
        columns=X.columns
    )

    for params in param_combinations:
        current_params = params.copy()
        all_folds_metrics = {metric: [] for metric in eval_metric}

        for train_index, test_index in kf.split(X_scaled, Y):
            X_train, X_test = X_scaled.iloc[train_index], X_scaled.iloc[test_index]
            Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

            rep_metrics = {metric: [] for metric in eval_metric}

            for _ in range(CV_reps):
                # --- Build smooth terms for all continuous predictors ---
                n_features = X_train.shape[1]
                terms = s(0)
                for i in range(1, n_features):
                    terms += s(i)

                model = LogisticGAM(
                    terms=terms,
                    lam=current_params.get("lam", 1),
                    n_splines=current_params.get("n_splines", 10),
                    max_iter=current_params.get("max_iter", 1000)
                )

                model.fit(X_train, Y_train)

                Y_prob = model.predict_mu(X_test)
                Y_pred = (Y_prob > 0.5).astype(int)

                # --- Metrics ---
                if 'auc' in eval_metric:
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
                if 'PPV' in eval_metric or 'NPV' in eval_metric:
                    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
                    PPV = tp / (tp + fp) if (tp + fp) > 0 else np.nan
                    NPV = tn / (tn + fn) if (tn + fn) > 0 else np.nan
                    if 'PPV' in eval_metric:
                        rep_metrics['PPV'].append(PPV)
                    if 'NPV' in eval_metric:
                        rep_metrics['NPV'].append(NPV)
                if 'MCC' in eval_metric or 'mcc' in eval_metric:
                    mcc_val = matthews_corrcoef(Y_test, Y_pred)
                    if 'MCC' in eval_metric:
                        rep_metrics['MCC'].append(mcc_val)
                    if 'mcc' in eval_metric:
                        rep_metrics['mcc'].append(mcc_val)
                if 'balancedAcc' in eval_metric:
                    rep_metrics['balancedAcc'].append(balanced_accuracy_score(Y_test, Y_pred))
                if 'pr_auc' in eval_metric:
                    precision, recall, _ = precision_recall_curve(Y_test, Y_prob)
                    pr_auc = auc(recall, precision)
                    rep_metrics['pr_auc'].append(pr_auc)
                if any(m in eval_metric for m in ['tn', 'fp', 'fn', 'tp']):
                    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
                    if 'tn' in eval_metric: rep_metrics.setdefault('tn', []).append(tn)
                    if 'fp' in eval_metric: rep_metrics.setdefault('fp', []).append(fp)
                    if 'fn' in eval_metric: rep_metrics.setdefault('fn', []).append(fn)
                    if 'tp' in eval_metric: rep_metrics.setdefault('tp', []).append(tp)

            # Mean over CV_reps for this fold
            fold_mean_metrics = {metric: np.nanmean(values) for metric, values in rep_metrics.items()}
            for metric in eval_metric:
                all_folds_metrics[metric].append(fold_mean_metrics.get(metric, np.nan))

        # Mean over folds
        mean_over_folds = {metric: np.nanmean(values) for metric, values in all_folds_metrics.items()}

        # Model selection
        if mean_over_folds.get(model_choice_metric, -np.inf) > best_model_choice_value:
            best_model_choice_value = mean_over_folds[model_choice_metric]
            best_model = model
            best_params = current_params
            best_scores = mean_over_folds

    return best_model, best_scores, best_params