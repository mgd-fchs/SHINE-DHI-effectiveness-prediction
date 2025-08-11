# Standard libraries
import os
import warnings
import glob

# Progress bar
from tqdm import tqdm

# Data manipulation
import numpy as np
import pandas as pd

# Statistical analysis
from scipy import stats as st
from sklearn.utils import resample
import shap

# Preprocessing and model evaluation
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, recall_score,
    precision_recall_curve, confusion_matrix, auc, precision_score,
    balanced_accuracy_score, matthews_corrcoef
)

# Serialization
import joblib
from pre_processing import *
from plotting import *

# Suppress warnings
warnings.filterwarnings('ignore')


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

def test_oos(test_df, res_dir, best_model, best_params, target_var='responsive', plot=True):
    
    # Drop common variables
    test_features = test_df.drop(columns=[target_var] + ['id'])
    test_labels = test_df[target_var]
    
    # Make predictions
    y_pred_proba = best_model.predict_proba(test_features)[:, 1]  # Probabilities for the positive class
    y_pred = best_model.predict(test_features)
    
    # Compute metrics
    tn, fp, fn, tp = confusion_matrix(test_labels, y_pred).ravel()
    
    scores = {
        'auc': roc_auc_score(test_labels, y_pred_proba),
        'f1': f1_score(test_labels, y_pred),
        'accuracy': accuracy_score(test_labels, y_pred),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        'sensitivity': recall_score(test_labels, y_pred),
        'PPV': precision_score(test_labels, y_pred),
        'NPV': tn / (tn + fn) if (tn + fn) > 0 else np.nan,
        'MCC': matthews_corrcoef(test_labels, y_pred),
        'balancedAcc': balanced_accuracy_score(test_labels, y_pred),
        'pr_auc': roc_auc_score(test_labels, y_pred_proba),
        'tn': tn,
        'fn': fn,
        'tp': tp,
        'fp': fp
    }
    
    colors = ["#22223B", "#4A4E69", "#9A8C98", "#C9ADA7", "#F2E9E4"]

    # SHAP feature importance
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(test_features)
    shap_values = shap_values[:, :, 1]  # Extract SHAP values for positive class
    
    # Compute mean absolute SHAP values for importance ranking
    shap_importance = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({'feature': test_features.columns, 'importance': shap_importance})
    feature_importance = feature_importance.sort_values(by='importance', ascending=False)
    
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


def evaluate_top_models(res_dir, test_df, target_col='responsive', group_name="('group_sub',)", 
                        top_n=10, n_iterations=10, desired_positive_rate=0.24, plot=False, target_var='responsive'):
    """
    Loads top N models from a result directory, resamples the test set with a desired positive rate, 
    and evaluates each model over multiple iterations.

    Returns:
        summary_df: DataFrame with mean and 95% CI for each metric.
        all_scores_df: Raw scores from each resampling.
    """
    # Load test metrics to identify top models
    df_scores = pd.read_csv(os.path.join(res_dir, "all_test_scores.csv"))
    group_df = df_scores[df_scores['group'] == group_name]

    # Select top models based on AUC
    top_indices = group_df['auc'].nlargest(top_n).index.tolist()

    # Load corresponding models from saved files
    top_models = []
    model_dir = os.path.join(res_dir, "top10_group_sub_models")

    for i in range(1, top_n + 1):
        pattern = os.path.join(model_dir, f"model_rank{i}_*.joblib")
        matched_files = glob.glob(pattern)
        if matched_files:
            top_models.append(joblib.load(matched_files[0]))

    if len(top_models) == 0:
        raise ValueError(f"No models found")

    # Drop missing values
    test_df = test_df.dropna()

    # Split positives and negatives
    positive_cases = test_df[test_df[target_col] == 1]
    negative_cases = test_df[test_df[target_col] == 0]

    all_scores = []

    for model in tqdm(top_models, desc="Evaluating top models"):
        for _ in range(n_iterations):
            # Stratified resampling
            total_samples = len(test_df)
            n_pos = int(total_samples * desired_positive_rate)
            n_neg = total_samples - n_pos

            pos_sample = resample(positive_cases, replace=True, n_samples=n_pos, random_state=None)
            neg_sample = resample(negative_cases, replace=False, n_samples=n_neg, random_state=None)

            balanced_df = pd.concat([pos_sample, neg_sample]).sample(frac=1).reset_index(drop=True)

            scores, _ = test_oos(balanced_df, res_dir, model, [], plot=plot)
            all_scores.append(scores)

    scores_df = pd.DataFrame(all_scores)

    # Compute summary
    mean_scores = scores_df.mean()
    ci_lower, ci_upper = st.t.interval(0.95, df=len(scores_df)-1, loc=mean_scores, scale=scores_df.sem())

    summary_df = pd.DataFrame({
        'Mean': mean_scores,
        '95% CI Lower': ci_lower,
        '95% CI Upper': ci_upper
    })

    return summary_df, scores_df

