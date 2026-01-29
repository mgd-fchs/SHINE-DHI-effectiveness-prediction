# Standard library
import os
import glob
import warnings

# Third-party
import numpy as np
import pandas as pd
from tqdm import tqdm

from scipy.stats import norm

from sklearn.utils import resample
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    f1_score,
    recall_score,
    precision_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    balanced_accuracy_score,
    matthews_corrcoef,
    auc,
)

from statsmodels.stats.proportion import proportion_confint

import shap
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

# Local modules
from pre_processing import *
from plotting import *

# Suppress warnings
warnings.filterwarnings("ignore")


def delong_roc_variance(ground_truth, predictions):
    """Compute the DeLong variance for ROC AUC (Hanley & McNeil 1982 extension)."""
    # Sort by predicted score
    order = np.argsort(-predictions)
    predictions, ground_truth = predictions[order], ground_truth[order]
    distinct_value_indices = np.where(np.diff(predictions))[0]
    thresholds = np.r_[distinct_value_indices, ground_truth.size - 1]

    tpr = np.cumsum(ground_truth)[thresholds] / ground_truth.sum()
    fpr = (1 + thresholds - np.cumsum(ground_truth)[thresholds]) / (ground_truth.size - ground_truth.sum())

    V10 = tpr
    V01 = fpr
    auc = roc_auc_score(ground_truth, predictions)
    s10 = np.var(V10, ddof=1)
    s01 = np.var(V01, ddof=1)
    auc_var = (s10 / ground_truth.sum()) + (s01 / (ground_truth.size - ground_truth.sum()))
    return auc, auc_var


def analytic_cis(y_true, y_score, threshold=0.5, alpha=0.05):
    """
    Compute analytic CIs where feasible (AUC, sensitivity, specificity, PPV, NPV).
    """
    # --- AUC (DeLong) approximation fallback via Hanley & McNeil large-sample formula ---
    auc = roc_auc_score(y_true, y_score)
    n1 = np.sum(y_true == 1)
    n0 = np.sum(y_true == 0)
    Q1 = auc / (2 - auc)
    Q2 = 2 * auc**2 / (1 + auc)
    se_auc = np.sqrt((auc * (1 - auc) + (n1 - 1) * (Q1 - auc**2) + (n0 - 1) * (Q2 - auc**2)) / (n0 * n1))
    z = norm.ppf(1 - alpha / 2)
    auc_ci = (max(0, auc - z * se_auc), min(1, auc + z * se_auc))

    # --- Confusion matrix based metrics ---
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    def safe_prop_ci(x, n):
        if n == 0:
            return (np.nan, np.nan)
        return proportion_confint(x, n, alpha=alpha, method="wilson")

    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv  = tn / (tn + fn) if (tn + fn) > 0 else np.nan

    sens_ci = safe_prop_ci(tp, tp + fn)
    spec_ci = safe_prop_ci(tn, tn + fp)
    ppv_ci  = safe_prop_ci(tp, tp + fp)
    npv_ci  = safe_prop_ci(tn, tn + fn)

    return {
        "auc": auc, "auc_CI_lower": auc_ci[0], "auc_CI_upper": auc_ci[1],
        "sensitivity": sens, "sens_CI_lower": sens_ci[0], "sens_CI_upper": sens_ci[1],
        "specificity": spec, "spec_CI_lower": spec_ci[0], "spec_CI_upper": spec_ci[1],
        "PPV": ppv, "PPV_CI_lower": ppv_ci[0], "PPV_CI_upper": ppv_ci[1],
        "NPV": npv, "NPV_CI_lower": npv_ci[0], "NPV_CI_upper": npv_ci[1],
    }


def compute_test_metrics(Y_test, test_predictions, proba_predictions, test_scores):
    Y_test_flat = Y_test.ravel()
    
    test_scores['auc'].append(roc_auc_score(Y_test_flat, proba_predictions))
    test_scores['f1'].append(f1_score(Y_test_flat, test_predictions))
    test_scores['accuracy'].append(accuracy_score(Y_test_flat, test_predictions))

    tn, fp, fn, tp = confusion_matrix(Y_test_flat, test_predictions).ravel()
    test_scores['specificity'].append(tn / (tn + fp) if (tn + fp) > 0 else np.nan)
    test_scores['sensitivity'].append(tp / (tp + fn) if (tp + fn) > 0 else np.nan)
    test_scores['PPV'].append(tp / (tp + fp) if (tp + fp) > 0 else np.nan)
    test_scores['NPV'].append(tn / (tn + fn) if (tn + fn) > 0 else np.nan)
    test_scores['MCC'].append(matthews_corrcoef(Y_test_flat, test_predictions))
    test_scores['balancedAcc'].append(balanced_accuracy_score(Y_test_flat, test_predictions))

    precision, recall, _ = precision_recall_curve(Y_test_flat, proba_predictions)
    pr_auc = auc(recall, precision)
    test_scores['pr_auc'].append(pr_auc)

    test_scores['tn'].append(tn)
    test_scores['fp'].append(fp)
    test_scores['fn'].append(fn)
    test_scores['tp'].append(tp)

    return test_scores

def flatten_score_dict(score_dict, res_dir=None, filename=None):
    rows = []
    for combination, metrics in score_dict.items():
        row = {"Combination": combination}
        for metric, values in metrics.items():
            row[f"{metric}_mean"] = values["mean"]
            row[f"{metric}_CI_lower"] = values["95%_CI"][0]
            row[f"{metric}_CI_upper"] = values["95%_CI"][1]
        rows.append(row)

    df = pd.DataFrame(rows)
    df_comb = pd.DataFrame(df["Combination"].tolist(), columns=[f"Factor_{i+1}" for i in range(df["Combination"].map(len).max())])
    df = pd.concat([df_comb, df.drop(columns="Combination")], axis=1)

    if res_dir and filename:
        df.to_csv(f"{res_dir}/{filename}", index=False)

    return df

def test_oos(
    test_df, 
    res_dir, 
    best_model, 
    best_params, 
    target_var='responsive', 
    plot=True,
    n_perm_repeats=1000,
    random_state=321
):
    """
    Evaluate model on out-of-sample data and compute permutation-based p-values for all metrics.
    """

    rng = np.random.default_rng(random_state)

    # Drop common variables
    test_features = test_df.drop(columns=[target_var, 'id'], errors='ignore')
    test_labels = test_df[target_var].to_numpy()

    # Make predictions
    y_pred_proba = best_model.predict_proba(test_features)[:, 1]
    y_pred = best_model.predict(test_features)

    output_df = pd.concat(
    [test_df[['id']].reset_index(drop=True), 
     test_features.reset_index(drop=True), 
     pd.Series(y_pred_proba, name='predicted_probability')], 
    axis=1
    )

    # Save to CSV
    output_df.to_csv('./predicted_probabilities_study2.csv', index=False)

    # Compute metrics
    tn, fp, fn, tp = confusion_matrix(test_labels, y_pred, labels=[0, 1]).ravel()

    scores = {
        'auc': roc_auc_score(test_labels, y_pred_proba),
        'f1': f1_score(test_labels, y_pred),
        'accuracy': accuracy_score(test_labels, y_pred),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        'sensitivity': recall_score(test_labels, y_pred),
        'PPV': precision_score(test_labels, y_pred) if (tp + fp) > 0 else np.nan,
        'NPV': tn / (tn + fn) if (tn + fn) > 0 else np.nan,
        'MCC': matthews_corrcoef(test_labels, y_pred) if (tp+tn+fp+fn) > 0 else np.nan,
        'balancedAcc': balanced_accuracy_score(test_labels, y_pred),
        'pr_auc': average_precision_score(test_labels, y_pred_proba),
        'tn': tn, 'fn': fn, 'tp': tp, 'fp': fp
    }

    # --- Permutation tests for each metric ---
    metrics = ['auc', 'f1', 'accuracy', 'specificity', 'sensitivity',
               'PPV', 'NPV', 'MCC', 'balancedAcc', 'pr_auc']
    # perm_p = {}

    # for metric in metrics:
    #     obs_val = scores[metric]
    #     if np.isnan(obs_val):
    #         perm_p[metric] = np.nan
    #         continue

    #     perm_vals = []
    #     for _ in range(n_perm_repeats):
    #         perm_labels = rng.permutation(test_labels)
    #         try:
    #             if metric == 'auc':
    #                 val = roc_auc_score(perm_labels, y_pred_proba)
    #             elif metric == 'pr_auc':
    #                 val = average_precision_score(perm_labels, y_pred_proba)
    #             else:
    #                 tn_p, fp_p, fn_p, tp_p = confusion_matrix(perm_labels, y_pred, labels=[0, 1]).ravel()
    #                 if metric == 'f1':
    #                     val = f1_score(perm_labels, y_pred)
    #                 elif metric == 'accuracy':
    #                     val = accuracy_score(perm_labels, y_pred)
    #                 elif metric == 'specificity':
    #                     val = tn_p / (tn_p + fp_p) if (tn_p + fp_p) > 0 else np.nan
    #                 elif metric == 'sensitivity':
    #                     val = tp_p / (tp_p + fn_p) if (tp_p + fn_p) > 0 else np.nan
    #                 elif metric == 'PPV':
    #                     val = tp_p / (tp_p + fp_p) if (tp_p + fp_p) > 0 else np.nan
    #                 elif metric == 'NPV':
    #                     val = tn_p / (tn_p + fn_p) if (tn_p + fn_p) > 0 else np.nan
    #                 elif metric == 'MCC':
    #                     val = matthews_corrcoef(perm_labels, y_pred)
    #                 elif metric == 'balancedAcc':
    #                     val = balanced_accuracy_score(perm_labels, y_pred)
    #                 else:
    #                     val = np.nan
    #         except ValueError:
    #             val = np.nan
    #         perm_vals.append(val)

    auc, auc_var = delong_roc_variance(test_labels, y_pred_proba)
    auc_std = np.sqrt(auc_var)

    z = (auc - 0.5) / auc_std
    p_value = 1 - norm.cdf(z)        # one-sided test: AUC > 0.5

    ci_lower = auc - 1.96 * auc_std
    ci_upper = auc + 1.96 * auc_std

    print(f"AUC = {auc:.3f}")
    print(f"95% CI (DeLong): [{ci_lower:.3f}, {ci_upper:.3f}]")
    print(f"p (AUC > 0.5) = {p_value:.4f}")

        # perm_vals = np.array([v for v in perm_vals if not np.isnan(v)])
        # if len(perm_vals) == 0:
        #     perm_p[metric] = np.nan
        # else:
        #     perm_p[metric] = (np.sum(perm_vals >= obs_val) + 1) / (len(perm_vals) + 1)

    # Add permutation p-values to scores
    # for metric in metrics:
    #     scores[f"{metric}_p_perm"] = perm_p.get(metric, np.nan)

    # --- SHAP feature importance ---
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(test_features)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]  # handle multi-output models

    shap_importance = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({
        'feature': test_features.columns,
        'importance': shap_importance
    }).sort_values(by='importance', ascending=False)

    # --- Plotting ---
    if plot:
        plot_shap_summary_with_percentages(
            all_shap_values=[shap_values],
            all_test_data=[pd.DataFrame(test_features)],
            res_dir=res_dir,
            combo="test_oos"
        )
        plot_pdp_across_runs(
            best_model=best_model,
            res_dir=res_dir,
            all_test_data=[pd.DataFrame(test_features)],
            interaction_pair=("avg_alcmost_freq", "avg_alcmost"),
            title="study_2"
        )

    return scores, best_params


def test_oos_gam(
    test_df, 
    res_dir, 
    best_model, 
    best_params, 
    target_var='responsive', 
    plot=True,
    n_perm_repeats=1000,
    random_state=321
):
    """
    Evaluate LogisticGAM on out-of-sample data and compute permutation-based p-values for all metrics.
    Includes GAM-specific term importance and partial dependence plots.
    """

    rng = np.random.default_rng(random_state)

    # Drop common variables
    test_features = test_df.drop(columns=[target_var, 'id'], errors='ignore')
    test_labels = test_df[target_var].to_numpy()

    # Predictions
    y_pred_proba = best_model.predict_mu(test_features)
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(test_labels, y_pred, labels=[0, 1]).ravel()

    scores = {
        'auc': roc_auc_score(test_labels, y_pred_proba),
        'f1': f1_score(test_labels, y_pred),
        'accuracy': accuracy_score(test_labels, y_pred),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        'sensitivity': recall_score(test_labels, y_pred),
        'PPV': precision_score(test_labels, y_pred) if (tp + fp) > 0 else np.nan,
        'NPV': tn / (tn + fn) if (tn + fn) > 0 else np.nan,
        'MCC': matthews_corrcoef(test_labels, y_pred) if (tp+tn+fp+fn) > 0 else np.nan,
        'balancedAcc': balanced_accuracy_score(test_labels, y_pred),
        'pr_auc': average_precision_score(test_labels, y_pred_proba),
        'tn': tn, 'fn': fn, 'tp': tp, 'fp': fp
    }

    # --- Permutation tests ---
    metrics = ['auc', 'f1', 'accuracy', 'specificity', 'sensitivity',
               'PPV', 'NPV', 'MCC', 'balancedAcc', 'pr_auc']
    perm_p = {}

    for metric in metrics:
        obs_val = scores[metric]
        if np.isnan(obs_val):
            perm_p[metric] = np.nan
            continue

        perm_vals = []
        for _ in range(n_perm_repeats):
            perm_labels = rng.permutation(test_labels)
            try:
                if metric == 'auc':
                    val = roc_auc_score(perm_labels, y_pred_proba)
                elif metric == 'pr_auc':
                    val = average_precision_score(perm_labels, y_pred_proba)
                else:
                    tn_p, fp_p, fn_p, tp_p = confusion_matrix(perm_labels, y_pred, labels=[0, 1]).ravel()
                    if metric == 'f1':
                        val = f1_score(perm_labels, y_pred)
                    elif metric == 'accuracy':
                        val = accuracy_score(perm_labels, y_pred)
                    elif metric == 'specificity':
                        val = tn_p / (tn_p + fp_p) if (tn_p + fp_p) > 0 else np.nan
                    elif metric == 'sensitivity':
                        val = tp_p / (tp_p + fn_p) if (tp_p + fn_p) > 0 else np.nan
                    elif metric == 'PPV':
                        val = tp_p / (tp_p + fp_p) if (tp_p + fp_p) > 0 else np.nan
                    elif metric == 'NPV':
                        val = tn_p / (tn_p + fn_p) if (tn_p + fn_p) > 0 else np.nan
                    elif metric == 'MCC':
                        val = matthews_corrcoef(perm_labels, y_pred)
                    elif metric == 'balancedAcc':
                        val = balanced_accuracy_score(perm_labels, y_pred)
                    else:
                        val = np.nan
            except ValueError:
                val = np.nan
            perm_vals.append(val)

        perm_vals = np.array([v for v in perm_vals if not np.isnan(v)])
        if len(perm_vals) == 0:
            perm_p[metric] = np.nan
        else:
            perm_p[metric] = (np.sum(perm_vals >= obs_val) + 1) / (len(perm_vals) + 1)

    for metric in metrics:
        scores[f"{metric}_p_perm"] = perm_p.get(metric, np.nan)

    # --- GAM feature importance ---
    try:
        term_importance = np.abs(best_model.statistics_['edof'])  # effective degrees of freedom per term
        feature_importance = pd.DataFrame({
            'feature': test_features.columns,
            'importance': term_importance
        }).sort_values(by='importance', ascending=False)
    except Exception:
        feature_importance = pd.DataFrame({'feature': test_features.columns, 'importance': np.nan})

    # --- Plotting ---
    if plot:
        # Partial dependence for each term
        for i, feature in enumerate(test_features.columns):
            XX = best_model.generate_X_grid(term=i)
            plt.figure(figsize=(5, 4))
            plt.plot(XX[:, i], best_model.partial_dependence(term=i, X=XX))
            plt.title(f"{feature} — Partial dependence")
            plt.xlabel(feature)
            plt.ylabel("Partial effect")
            plt.tight_layout()
            plt.savefig(f"{res_dir}/pdp_{feature}.png", dpi=300)
            plt.close()

    return scores, best_params, feature_importance



def test_oos_logreg(
    test_df, 
    res_dir, 
    best_model, 
    best_params=None, 
    target_var='responsive', 
    plot=True,
    n_perm_repeats=1000,
    random_state=321
):
    """
    Evaluate logistic regression model (possibly wrapped in a pipeline) on out-of-sample data.
    Computes permutation-based p-values for each metric.
    """

    rng = np.random.default_rng(random_state)

    # Prepare data
    test_features = test_df.drop(columns=[target_var, 'id'], errors='ignore')
    test_labels = test_df[target_var].to_numpy()

    # Predictions
    y_pred_proba = best_model.predict_proba(test_features)[:, 1]
    y_pred = best_model.predict(test_features)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(test_labels, y_pred, labels=[0,1]).ravel()

    # Compute metrics safely
    scores = {
        'auc': roc_auc_score(test_labels, y_pred_proba),
        'f1': f1_score(test_labels, y_pred, zero_division=0),
        'accuracy': accuracy_score(test_labels, y_pred),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        'sensitivity': recall_score(test_labels, y_pred, zero_division=0),
        'PPV': precision_score(test_labels, y_pred, zero_division=0),
        'NPV': tn / (tn + fn) if (tn + fn) > 0 else np.nan,
        'MCC': matthews_corrcoef(test_labels, y_pred) if len(np.unique(y_pred)) > 1 else np.nan,
        'balancedAcc': balanced_accuracy_score(test_labels, y_pred),
        'pr_auc': average_precision_score(test_labels, y_pred_proba),
        'tn': tn,
        'fn': fn,
        'tp': tp,
        'fp': fp
    }

    # --- Permutation tests for each metric ---
    metrics = ['auc', 'f1', 'accuracy', 'specificity', 'sensitivity',
               'PPV', 'NPV', 'MCC', 'balancedAcc', 'pr_auc']
    # perm_p = {}

    # for metric in metrics:
    #     obs_val = scores[metric]
    #     if np.isnan(obs_val):
    #         perm_p[metric] = np.nan
    #         continue

    #     perm_vals = []
    #     for _ in range(n_perm_repeats):
    #         perm_labels = rng.permutation(test_labels)
    #         try:
    #             if metric == 'auc':
    #                 val = roc_auc_score(perm_labels, y_pred_proba)
    #             elif metric == 'pr_auc':
    #                 val = average_precision_score(perm_labels, y_pred_proba)
    #             else:
    #                 tn_p, fp_p, fn_p, tp_p = confusion_matrix(perm_labels, y_pred, labels=[0,1]).ravel()
    #                 if metric == 'f1':
    #                     val = f1_score(perm_labels, y_pred, zero_division=0)
    #                 elif metric == 'accuracy':
    #                     val = accuracy_score(perm_labels, y_pred)
    #                 elif metric == 'specificity':
    #                     val = tn_p / (tn_p + fp_p) if (tn_p + fp_p) > 0 else np.nan
    #                 elif metric == 'sensitivity':
    #                     val = tp_p / (tp_p + fn_p) if (tp_p + fn_p) > 0 else np.nan
    #                 elif metric == 'PPV':
    #                     val = tp_p / (tp_p + fp_p) if (tp_p + fp_p) > 0 else np.nan
    #                 elif metric == 'NPV':
    #                     val = tn_p / (tn_p + fn_p) if (tn_p + fn_p) > 0 else np.nan
    #                 elif metric == 'MCC':
    #                     val = matthews_corrcoef(perm_labels, y_pred)
    #                 elif metric == 'balancedAcc':
    #                     val = balanced_accuracy_score(perm_labels, y_pred)
    #                 else:
    #                     val = np.nan
    #         except ValueError:
    #             val = np.nan
    #         perm_vals.append(val)

        # perm_vals = np.array([v for v in perm_vals if not np.isnan(v)])
        # if len(perm_vals) == 0:
        #     perm_p[metric] = np.nan
        # else:
        #     perm_p[metric] = (np.sum(perm_vals >= obs_val) + 1) / (len(perm_vals) + 1)

    # # Add permutation p-values to scores
    # for metric in metrics:
    #     scores[f"{metric}_p_perm"] = perm_p.get(metric, np.nan)
    auc, auc_var = delong_roc_variance(test_labels, y_pred_proba)
    auc_std = np.sqrt(auc_var)

    z = (auc - 0.5) / auc_std
    p_value = 1 - norm.cdf(z)        # one-sided test: AUC > 0.5

    ci_lower = auc - 1.96 * auc_std
    ci_upper = auc + 1.96 * auc_std

    print(f"AUC = {auc:.3f}")
    print(f"95% CI (DeLong): [{ci_lower:.3f}, {ci_upper:.3f}]")
    print(f"p (AUC > 0.5) = {p_value:.4f}")

    # --- Extract coefficients ---
    if hasattr(best_model, "named_steps") and "logisticregression" in best_model.named_steps:
        logreg_step = best_model.named_steps["logisticregression"]
    else:
        logreg_step = best_model

    coefs = logreg_step.coef_.flatten()
    odds_ratios = np.exp(coefs)
    abs_coefs = np.abs(coefs)

    feature_importance = pd.DataFrame({
        "feature": test_features.columns,
        "coef": coefs,
        "abs_coef": abs_coefs,
        "odds_ratio": odds_ratios
    }).sort_values(by="abs_coef", ascending=False)

    # Optional plotting
    if plot:
        topn = min(20, len(feature_importance))
        plt.figure(figsize=(8, 6))
        plt.barh(
            feature_importance["feature"][:topn][::-1],
            feature_importance["coef"][:topn][::-1],
            color=np.where(feature_importance["coef"][:topn][::-1] > 0, "#4A4E69", "#C9ADA7")
        )
        plt.xlabel("Coefficient")
        plt.title("Top {} Logistic Regression Coefficients".format(topn))
        plt.tight_layout()
        plt.show()

    return scores, best_params, feature_importance



def test_oos_svm(
    test_df,
    res_dir,
    best_model,
    best_params=None,
    target_var='responsive',
    plot=True,
    n_perm_repeats=1000,   # number of permutations
    random_state=321
):
    """
    Evaluate SVM model on out-of-sample data and compute permutation-based p-values.

    For each metric, p = fraction of label permutations that achieve an equal or higher score.
    """

    rng = np.random.default_rng(random_state)

    # Prepare features/labels
    test_features = test_df.drop(columns=[target_var, 'id'], errors='ignore')
    test_labels = test_df[target_var].to_numpy()

    # Predictions
    if hasattr(best_model, "predict_proba"):
        y_score = best_model.predict_proba(test_features)[:, 1]
    elif hasattr(best_model, "decision_function"):
        y_score = best_model.decision_function(test_features)
        if y_score.ndim > 1:
            y_score = y_score[:, 1]
    else:
        y_score = None

    y_pred = best_model.predict(test_features)

    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(test_labels, y_pred, labels=[0, 1]).ravel()

    # Performance metrics
    scores = {
        'auc': roc_auc_score(test_labels, y_score) if y_score is not None else np.nan,
        'f1': f1_score(test_labels, y_pred),
        'accuracy': accuracy_score(test_labels, y_pred),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        'sensitivity': recall_score(test_labels, y_pred),
        'PPV': precision_score(test_labels, y_pred) if (tp + fp) > 0 else np.nan,
        'NPV': tn / (tn + fn) if (tn + fn) > 0 else np.nan,
        'MCC': matthews_corrcoef(test_labels, y_pred) if (tp + tn + fp + fn) > 0 else np.nan,
        'balancedAcc': balanced_accuracy_score(test_labels, y_pred),
        'pr_auc': average_precision_score(test_labels, y_score) if y_score is not None else np.nan,
        'tn': tn, 'fn': fn, 'tp': tp, 'fp': fp
    }

    # --- Permutation tests for each metric ---
    metrics = ['auc', 'f1', 'accuracy', 'specificity', 'sensitivity',
               'PPV', 'NPV', 'MCC', 'balancedAcc', 'pr_auc']
    perm_p = {}

    for metric in metrics:
        obs_val = scores[metric]
        if np.isnan(obs_val):
            perm_p[metric] = np.nan
            continue

    #     perm_vals = []
    #     for _ in range(n_perm_repeats):
    #         perm_labels = rng.permutation(test_labels)

    #         try:
    #             if metric == 'auc' and y_score is not None:
    #                 val = roc_auc_score(perm_labels, y_score)
    #             elif metric == 'pr_auc' and y_score is not None:
    #                 val = average_precision_score(perm_labels, y_score)
    #             else:
    #                 # Recompute confusion-matrix-based metric
    #                 tn_p, fp_p, fn_p, tp_p = confusion_matrix(
    #                     perm_labels, y_pred, labels=[0, 1]
    #                 ).ravel()
    #                 if metric == 'f1':
    #                     val = f1_score(perm_labels, y_pred)
    #                 elif metric == 'accuracy':
    #                     val = accuracy_score(perm_labels, y_pred)
    #                 elif metric == 'specificity':
    #                     val = tn_p / (tn_p + fp_p) if (tn_p + fp_p) > 0 else np.nan
    #                 elif metric == 'sensitivity':
    #                     val = tp_p / (tp_p + fn_p) if (tp_p + fn_p) > 0 else np.nan
    #                 elif metric == 'PPV':
    #                     val = tp_p / (tp_p + fp_p) if (tp_p + fp_p) > 0 else np.nan
    #                 elif metric == 'NPV':
    #                     val = tn_p / (tn_p + fn_p) if (tn_p + fn_p) > 0 else np.nan
    #                 elif metric == 'MCC':
    #                     val = matthews_corrcoef(perm_labels, y_pred)
    #                 elif metric == 'balancedAcc':
    #                     val = balanced_accuracy_score(perm_labels, y_pred)
    #                 else:
    #                     val = np.nan
    #         except ValueError:
    #             val = np.nan
    #         perm_vals.append(val)

    #     perm_vals = np.array([v for v in perm_vals if not np.isnan(v)])
    #     if len(perm_vals) == 0:
    #         perm_p[metric] = np.nan
    #     else:
    #         perm_p[metric] = (np.sum(perm_vals >= obs_val) + 1) / (len(perm_vals) + 1)

    # # Add permutation p-values to scores
    # for metric in metrics:
    #     scores[f"{metric}_p_perm"] = perm_p.get(metric, np.nan)

    auc, auc_var = delong_roc_variance(test_labels, y_score)
    auc_std = np.sqrt(auc_var)

    z = (auc - 0.5) / auc_std
    p_value = 1 - norm.cdf(z)        # one-sided test: AUC > 0.5

    ci_lower = auc - 1.96 * auc_std
    ci_upper = auc + 1.96 * auc_std

    print(f"AUC = {auc:.3f}")
    print(f"95% CI (DeLong): [{ci_lower:.3f}, {ci_upper:.3f}]")
    print(f"p (AUC > 0.5) = {p_value:.4f}")

    return scores, best_params, None


def evaluate_top_models(
    res_dir,
    test_df,
    target_col='responsive',
    group_name="('group_sub',)",
    top_n=10,
    n_iterations=10,
    desired_positive_rate=None,  # None = keep observed prevalence
    m=None,                      # optional subset size (m-out-of-n bootstrap)
    plot=False,
    target_var='responsive',
    n_permutations=100           # number of label permutations (global)
):
    """
    Bootstrapped evaluation of top models on a fixed test set, 
    with global permutation-based p-values for each metric.

    Bootstrapping estimates uncertainty; a global permutation test
    estimates significance against chance performance.
    """
    # Load test metrics to identify top models
    df_scores = pd.read_csv(os.path.join(res_dir, "all_test_scores.csv"))
    group_df = df_scores[df_scores['group'] == group_name]

    # Select top models based on AUC
    top_indices = group_df['auc'].nlargest(top_n).index.tolist()

    # Load corresponding models
    model_dir = os.path.join(res_dir, "top100_group_sub_models")
    top_models = []
    for i in range(1, top_n + 1):
        pattern = os.path.join(model_dir, f"model_rank{i}_*.joblib")
        matched_files = glob.glob(pattern)
        if matched_files:
            top_models.append(joblib.load(matched_files[0]))

    if len(top_models) == 0:
        raise ValueError("No models found")

    # Drop missing values
    test_df = test_df.dropna(subset=[target_col])

    # Split positives and negatives
    positive_cases = test_df[test_df[target_col] == 1]
    negative_cases = test_df[test_df[target_col] == 0]
    total_samples = len(test_df)

    if m is None:
        m = total_samples  # full-size bootstrap

    all_scores = []

    # --- Bootstrapping for uncertainty estimation ---
    for model in tqdm(top_models, desc="Evaluating top models"):
        for _ in range(n_iterations):
            # # Determine sampling ratio
            # if desired_positive_rate is None:
            #     p_rate = len(positive_cases) / total_samples
            # else:
            #     p_rate = desired_positive_rate

            # n_pos = int(m * p_rate)
            # n_neg = m - n_pos

            # # Bootstrap with replacement for both classes
            # pos_sample = resample(positive_cases, replace=True, n_samples=n_pos, random_state=None)
            # neg_sample = resample(negative_cases, replace=True, n_samples=n_neg, random_state=None)

            # balanced_df = pd.concat([pos_sample, neg_sample]).sample(frac=1).reset_index(drop=True)

            balanced_df = test_df

            # Evaluate on resampled test set
            scores, _ = test_oos(balanced_df, res_dir, model, [], plot=plot)
            all_scores.append(scores)

    scores_df = pd.DataFrame(all_scores)

    # --- Global permutation test on the original test set ---
    perm_distributions = {metric: [] for metric in scores_df.columns}

    for _perm in tqdm(range(n_permutations), desc="Running global permutation test"):
        perm_df = test_df.copy()
        perm_df[target_col] = np.random.permutation(perm_df[target_col].values)
        for model in top_models:
            perm_scores, _ = test_oos(perm_df, res_dir, model, [], plot=False)
            for metric, val in perm_scores.items():
                if metric in perm_distributions:
                    perm_distributions[metric].append(val)

    # --- Summary for model performance ---
    summary_rows = []
    for metric in scores_df.columns:
        vals = scores_df[metric].to_numpy()
        mean = np.mean(vals)
        ci_lower, ci_upper = np.percentile(vals, [2.5, 97.5])

        # Compute permutation-based p-value (global)
        if metric in perm_distributions:
            perm_vals = np.array(perm_distributions[metric])
            p_value = (np.sum(perm_vals >= mean) + 1) / (len(perm_vals) + 1)
        else:
            p_value = np.nan

        summary_rows.append({
            'Metric': metric,
            'Mean': mean,
            '95% CI Lower': ci_lower,
            '95% CI Upper': ci_upper,
            'p (empirical)': p_value
        })

    summary_df = pd.DataFrame(summary_rows).set_index('Metric')

    # --- Separate summary for permuted (null) distributions ---
    perm_summary_rows = []
    for metric, vals in perm_distributions.items():
        vals = np.array(vals)
        mean = np.mean(vals)
        ci_lower, ci_upper = np.percentile(vals, [2.5, 97.5])
        perm_summary_rows.append({
            'Metric': metric,
            'Mean (perm)': mean,
            '95% CI Lower (perm)': ci_lower,
            '95% CI Upper (perm)': ci_upper
        })
    perm_summary_df = pd.DataFrame(perm_summary_rows).set_index('Metric')

    # --- Plot distributions for AUC, F1, and balanced accuracy ---
    import matplotlib.pyplot as plt
    import seaborn as sns

    metrics_to_plot = ['auc', 'f1', 'balancedAcc']
    for metric in metrics_to_plot:
        if metric in scores_df.columns and metric in perm_distributions:
            plt.figure(figsize=(5, 4))
            sns.boxplot(
                data=[
                    scores_df[metric].dropna().values,
                    np.array(perm_distributions[metric])
                ],
                orient="v"
            )
            plt.xticks([0, 1], ["Model", "Permuted"])
            plt.title(f"{metric.upper()} Distribution")
            plt.ylabel(metric.upper())
            plt.tight_layout()
            plt.show()

    return summary_df, perm_summary_df, scores_df

def evaluate_top_models_logreg(
    res_dir, 
    test_df, 
    target_col='responsive', 
    group_name="('group_sub',)", 
    top_n=10, 
    n_iterations=10, 
    desired_positive_rate=None,  # None = keep observed rate
    m=None,                      # optional subset size (m-out-of-n bootstrap)
    plot=False,
    n_permutations=100           # number of label permutations (global, not per bootstrap)
):
    """
    Bootstrapped evaluation of top logistic regression models on a fixed test set,
    with permutation-based p-values and FDR correction (simplified).

    Bootstrapping estimates uncertainty; a global permutation test estimates
    significance against chance performance.
    """

    # Load test metrics to identify top models
    df_scores = pd.read_csv(os.path.join(res_dir, "all_test_scores.csv"))
    group_df = df_scores[df_scores['group'] == group_name]

    # Select top models based on AUC
    top_indices = group_df['auc'].nlargest(top_n).index.tolist()

    # Load corresponding models
    model_dir = os.path.join(res_dir, "top100_group_sub_models")
    top_models = []
    for i in range(1, top_n + 1):
        pattern = os.path.join(model_dir, f"model_rank{i}_*.joblib")
        matched_files = glob.glob(pattern)
        if matched_files:
            top_models.append(joblib.load(matched_files[0]))

    if len(top_models) == 0:
        raise ValueError("No logistic regression models found in directory.")

    # Drop missing values
    test_df = test_df.dropna(subset=[target_col])

    # Split positives and negatives
    positive_cases = test_df[test_df[target_col] == 1]
    negative_cases = test_df[test_df[target_col] == 0]
    total_samples = len(test_df)

    if m is None:
        m = total_samples  # full-size bootstrap

    all_scores = []

    for model in tqdm(top_models, desc="Evaluating top logistic regression models"):
        for _ in range(n_iterations):
            # Determine target prevalence
            if desired_positive_rate is None:
                p_rate = len(positive_cases) / total_samples
            else:
                p_rate = desired_positive_rate

            n_pos = int(m * p_rate)
            n_neg = m - n_pos

            # Bootstrap with replacement for both classes
            pos_sample = resample(positive_cases, replace=True, n_samples=n_pos, random_state=None)
            neg_sample = resample(negative_cases, replace=True, n_samples=n_neg, random_state=None)

            balanced_df = pd.concat([pos_sample, neg_sample]).sample(frac=1, random_state=None).reset_index(drop=True)

            # Evaluate on resampled test set
            scores, _, _ = test_oos_logreg(balanced_df, res_dir, model, None, target_var=target_col, plot=plot)
            scores['model'] = str(model)
            all_scores.append(scores)

    # Combine results
    scores_df = pd.DataFrame(all_scores)

    # --- Global permutation test on the original test set ---
    perm_distributions = {metric: [] for metric in scores_df.select_dtypes(include=[np.number]).columns}

    for _perm in tqdm(range(n_permutations), desc="Running global permutation test"):
        perm_df = test_df.copy()
        perm_df[target_col] = np.random.permutation(perm_df[target_col].values)
        for model in top_models:
            perm_scores, _, _ = test_oos_logreg(perm_df, res_dir, model, None, target_var=target_col, plot=False)
            for metric, val in perm_scores.items():
                if metric in perm_distributions:
                    perm_distributions[metric].append(val)

    # Compute mean, CI, and permutation-based p-values
    summary_rows = []
    for metric in scores_df.select_dtypes(include=[np.number]).columns:
        vals = scores_df[metric].dropna().to_numpy()
        mean = np.mean(vals)
        ci_lower, ci_upper = np.percentile(vals, [2.5, 97.5])

        # Compute permutation-based p-value (global)
        if metric in perm_distributions:
            perm_vals = np.array(perm_distributions[metric])
            p_value = (np.sum(perm_vals >= mean) + 1) / (len(perm_vals) + 1)
        else:
            p_value = np.nan

        summary_rows.append({
            'Metric': metric,
            'Mean': mean,
            '95% CI Lower': ci_lower,
            '95% CI Upper': ci_upper,
            'p (uncorrected)': p_value
        })

    metrics_to_plot = ['auc', 'f1', 'balancedAcc']
    for metric in metrics_to_plot:
        if metric in scores_df.columns and metric in perm_distributions:
            plt.figure(figsize=(5, 4))
            sns.boxplot(
                data=[
                    scores_df[metric].dropna().values,
                    np.array(perm_distributions[metric])
                ],
                orient="v"
            )
            plt.xticks([0, 1], ["Model", "Permuted"])
            plt.title(f"{metric.upper()} Distribution")
            plt.ylabel(metric.upper())
            plt.tight_layout()
            plt.show()

    summary_df = pd.DataFrame(summary_rows).set_index('Metric')
    return summary_df, scores_df


def evaluate_top_models_svm(
    res_dir, 
    test_df, 
    target_col='responsive', 
    group_name="('group_sub',)", 
    top_n=10, 
    n_iterations=10, 
    desired_positive_rate=None,  # None = keep observed rate
    m=None,                      # optional subset size (m-out-of-n bootstrap)
    plot=False,
    n_permutations=100           # number of label permutations (global, not per bootstrap)
):
    """
    Bootstrapped evaluation of top SVM models on a fixed test set,
    with a single global permutation test for significance.
    Mirrors evaluate_top_models_logreg.
    """

    # Load test metrics to identify top models
    df_scores = pd.read_csv(os.path.join(res_dir, "all_test_scores.csv"))
    group_df = df_scores[df_scores['group'] == group_name]

    # Select top models based on AUC
    top_indices = group_df['auc'].nlargest(top_n).index.tolist()

    # Load corresponding models
    model_dir = os.path.join(res_dir, "top100_group_sub_models")
    top_models = []
    for i in range(1, top_n + 1):
        pattern = os.path.join(model_dir, f"model_rank{i}_*.joblib")
        matched_files = glob.glob(pattern)
        if matched_files:
            top_models.append(joblib.load(matched_files[0]))

    if len(top_models) == 0:
        raise ValueError("No SVM models found in directory.")

    # Drop missing values
    test_df = test_df.dropna(subset=[target_col])

    # Split positives and negatives
    positive_cases = test_df[test_df[target_col] == 1]
    negative_cases = test_df[test_df[target_col] == 0]
    total_samples = len(test_df)

    if m is None:
        m = total_samples  # full-size bootstrap

    all_scores = []

    # --- Bootstrapping for uncertainty estimation ---
    if desired_positive_rate:
        for model in tqdm(top_models, desc="Evaluating top SVM models"):
            for _ in range(n_iterations):
                # Keep all negatives as-is
                neg_sample = negative_cases.copy()

                # Determine how many positives to resample (with replacement)
                n_pos_resample = int(total_samples * desired_positive_rate)
                pos_sample = resample(
                    positive_cases,
                    replace=True,
                    n_samples=min(n_pos_resample, len(positive_cases)),
                    random_state=None
                )

                # Combine original positives with resampled ones (deduping optional)
                combined_pos = pd.concat([positive_cases, pos_sample], ignore_index=True)

                # Combine positives and negatives, shuffle
                boot_df = pd.concat([combined_pos, neg_sample], ignore_index=True)
                boot_df = boot_df.sample(frac=1, random_state=None).reset_index(drop=True)

                # Evaluate on bootstrapped test set
                scores, _, _ = test_oos_svm(
                    boot_df,
                    res_dir,
                    model,
                    None,
                    target_var=target_col,
                    plot=plot
                )
                scores['model'] = str(model)
                all_scores.append(scores)


    if  desired_positive_rate == None:
        # --- Bootstrapping for uncertainty estimation (single-model, standard bootstrap) ---
        for model in tqdm(top_models, desc="Evaluating top SVM models"):
            for _ in range(n_iterations):
                # # Standard nonparametric bootstrap over the entire test set
                # boot_df = resample(
                #     test_df,
                #     replace=True,
                #     n_samples=m if m is not None else len(test_df),
                #     random_state=None
                # )
                boot_df= test_df
                # Evaluate on resampled test set
                scores, _, _ = test_oos_svm(
                    boot_df,
                    res_dir,
                    model,
                    None,
                    target_var=target_col,
                    plot=plot
                )
                scores['model'] = str(model)
                all_scores.append(scores)


    # Combine results
    scores_df = pd.DataFrame(all_scores)

    # --- Global permutation test on the original test set ---
    perm_distributions = {metric: [] for metric in scores_df.select_dtypes(include=[np.number]).columns}

    for _perm in tqdm(range(n_permutations), desc="Running global permutation test"):
        perm_df = test_df.copy()
        perm_df[target_col] = np.random.permutation(perm_df[target_col].values)
        for model in top_models:
            # Replace with your actual evaluation function for SVM:
            # perm_scores, _, _ = test_oos_svm(perm_df, res_dir, model, None, target_var=target_col, plot=False)
            perm_scores, _, _ = test_oos_svm(perm_df, res_dir, model, None, target_var=target_col, plot=False)
            for metric, val in perm_scores.items():
                if metric in perm_distributions:
                    perm_distributions[metric].append(val)

    # Compute mean, CI, and permutation-based p-values
    summary_rows = []
    for metric in scores_df.select_dtypes(include=[np.number]).columns:
        vals = scores_df[metric].dropna().to_numpy()
        mean = np.mean(vals)
        ci_lower, ci_upper = np.percentile(vals, [2.5, 97.5])

        if metric in perm_distributions:
            perm_vals = np.array(perm_distributions[metric])
            p_value = (np.sum(perm_vals >= mean) + 1) / (len(perm_vals) + 1)
        else:
            p_value = np.nan

        summary_rows.append({
            'Metric': metric,
            'Mean': mean,
            '95% CI Lower': ci_lower,
            '95% CI Upper': ci_upper,
            'p (uncorrected)': p_value
        })

    # Optional: plots of model vs permuted distributions
    import matplotlib.pyplot as plt
    import seaborn as sns
    metrics_to_plot = ['auc', 'f1', 'balancedAcc']
    for metric in metrics_to_plot:
        if metric in scores_df.columns and metric in perm_distributions:
            plt.figure(figsize=(5, 4))
            sns.boxplot(
                data=[scores_df[metric].dropna().values, np.array(perm_distributions[metric])],
                orient="v"
            )
            plt.xticks([0, 1], ["Model", "Permuted"])
            plt.title(f"{metric.upper()} Distribution")
            plt.ylabel(metric.upper())
            plt.tight_layout()
            plt.show()

    summary_df = pd.DataFrame(summary_rows).set_index('Metric')
    return summary_df, scores_df