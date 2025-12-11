# Standard libraries
import os
import csv
import warnings
from scipy.interpolate import interp1d
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import shap

# Model interpretation
from sklearn.inspection import PartialDependenceDisplay
from pre_processing import *

# Suppress warnings
warnings.filterwarnings('ignore')


def save_metrics_to_csv(results_dict, results_dir, filename):

    os.makedirs(results_dir, exist_ok=True)


    # Define the output file path
    file_path = os.path.join(results_dir, filename)
    
    # Extract all metric names
    all_metrics = set()
    for metrics in results_dict.values():
        all_metrics.update(metrics.keys())
    
    # Sort metrics for consistency
    all_metrics = sorted(all_metrics)

    # Open CSV file for writing
    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        
        # Write header
        header = ["run", "group"] + all_metrics
        writer.writerow(header)

        # Write data
        for group, metrics in results_dict.items():
            num_runs = len(next(iter(metrics.values())))  # Get number of runs from first metric
            for run_idx in range(num_runs):
                row = [run_idx, str(group)]  # Start with run index and group name
                for metric in all_metrics:
                    value = metrics.get(metric, [np.nan] * num_runs)[run_idx]  # Handle missing values
                    row.append(value)
                writer.writerow(row)



def plot_pdp_across_runs(best_model, res_dir, all_test_data, feature_names=None, interaction_pair=None, colors=None, title=None, ids=[]):
    """
    Plots PDPs with mean and std across multiple test sets for each feature.
    Optionally adds an interaction plot.

    Parameters:
        best_model: trained model
        all_test_data: list of pd.DataFrames used for PDP evaluation
        feature_names: list of features to plot (default: all features in data)
        interaction_pair: tuple of two features to plot interaction PDP
        colors: optional color list
    """
    if colors is None:
        colors = ["#22223B", "#4A4E69", "#9A8C98", "#C9ADA7", "#F2E9E4"]

    final_test_data = pd.concat(all_test_data, ignore_index=True)
    print(final_test_data.columns)

    if feature_names is None:
        feature_names = final_test_data.columns.tolist()
    
    # Optional: map feature names to preferred display names
    name_mapping = {
        "avg_alcmost": "Perceived peer drinking amount",
        "groupAtt_alc": "Perceived peer drinking attitudes",
        "avg_alcmost_freq": "Perceived peer drinking frequency",
        "alc_norm_5_r": "Perceived peer drinking approval",
        "groupAtt_binge": "Perceived peer binge drinking attitudes"
    }
    display_names = [name_mapping.get(name, name) for name in feature_names]


    num_features = len(feature_names)
    num_plots = num_features + (1 if interaction_pair else 0)
    num_cols = 3
    num_rows = -(-num_plots // num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 4))
    axes = axes.flatten()

    for idx, feature_name in enumerate(feature_names):
        pdp_values = []
        feature_values_list = []

        for dat in all_test_data:
            ax_dummy = plt.figure().add_subplot()
            ax_dummy.set_visible(False)
            pdp_display = PartialDependenceDisplay.from_estimator(best_model, dat, [feature_name], ax=ax_dummy)
            plt.close(ax_dummy.figure)

            pdp_x = pdp_display.lines_[0][0].get_xdata()
            pdp_y = pdp_display.lines_[0][0].get_ydata()
            pdp_values.append(pdp_y)
            feature_values_list.append(pdp_x)

        common_feature_values = np.linspace(min(map(min, feature_values_list)),
                                            max(map(max, feature_values_list)), num=100)

        interpolated_pdp_values = []
        for i in range(len(pdp_values)):
            f_interp = interp1d(feature_values_list[i], pdp_values[i], kind="linear", fill_value="extrapolate")
            interpolated_pdp_values.append(f_interp(common_feature_values))

        pdp_values = np.array(interpolated_pdp_values)
        pdp_mean = np.mean(pdp_values, axis=0)
        pdp_std = np.std(pdp_values, axis=0)

        # --- Store PDP data ---
        pdp_df = pd.DataFrame({
            "feature": feature_name,
            "x": common_feature_values,
            "y_mean": pdp_mean,
            "y_std": pdp_std
        })

        # Append to CSV (create if not exists)
        out_path = os.path.join(res_dir, "pdp_data.csv")
        header = not os.path.exists(out_path)
        pdp_df.to_csv(out_path, mode="a", header=header, index=False)


        ax = axes[idx]
        ax.plot(common_feature_values, pdp_mean, label="Mean PDP", color=colors[0], lw=2)
        ax.fill_between(common_feature_values, pdp_mean - pdp_std, pdp_mean + pdp_std,
                        color=colors[2], alpha=0.5, label="Std Dev")
        ax.set_ylabel("Predicted Value")            
        ax.set_title(f"{display_names[idx]}")
        ax.legend()

    if interaction_pair:
        ax = axes[num_features]
        PartialDependenceDisplay.from_estimator(best_model, final_test_data,
                                                [interaction_pair], ax=ax)
        name_a = name_mapping.get(interaction_pair[0], interaction_pair[0])
        name_b = name_mapping.get(interaction_pair[1], interaction_pair[1])
        ax.set_title(f"{name_a} & {name_b}")


    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()

    # Build filename
    interaction_suffix = f"_{interaction_pair[0]}_{interaction_pair[1]}" if interaction_pair else ""

    res_dir = os.path.join(res_dir, 'img', 'PDP')
    os.makedirs(res_dir, exist_ok=True)
    
    if not title:
        filename = f"{res_dir}/pdp_plots{interaction_suffix}.png"
    else:
        filename = f"{res_dir}/pdp_plots{title}.png"
    # Save figure
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

def plot_shap_summary_with_percentages(all_shap_values, all_test_data, res_dir, combo):
    # Mapping of original to preferred variable names
    name_mapping = {
        "avg_alcmost": "Peer Perception: Drinking Amount",
        "groupAtt_alc": "Peer Attitudes: Alcohol",
        "avg_alcmost_freq": "Peer Perception: Drinking Frequency",
        "alc_norm_5_r": "Perceived Peer Pressure",
        "groupAtt_binge": "Peer Attitudes: Binges"
    }

    # Combine SHAP values and test data
    final_shap_values = np.vstack(all_shap_values)
    final_test_data = pd.concat(all_test_data, ignore_index=True)

    # Compute relative importance
    mean_abs_shap = np.abs(final_shap_values).mean(axis=0)
    rel_importance = 100 * mean_abs_shap / mean_abs_shap.sum()

    # Plot SHAP summary without showing
    plt.figure()
    shap.summary_plot(final_shap_values, final_test_data, show=False, cmap='winter')

    # Get current axis and y-tick labels
    ax = plt.gca()
    feature_names = [tick.get_text() for tick in ax.get_yticklabels()]

    # Map to preferred names if available
    mapped_feature_names = [name_mapping.get(name, name) for name in feature_names]

    # Use Index.get_loc instead of list
    col_index = final_test_data.columns
    feature_order = [col_index.get_loc(name) for name in feature_names]

    # Add percentage values to labels
    percent_labels = [f"{mapped_name} ({rel_importance[i]:.1f}%)"
                      for mapped_name, i in zip(mapped_feature_names, feature_order)]
    ax.set_yticklabels(percent_labels, fontsize=10)

    # Save updated plot
    plt.tight_layout()
    
    res_dir = os.path.join(res_dir, 'img', 'SHAP')
    os.makedirs(res_dir, exist_ok=True)

    plt.savefig(os.path.join(res_dir, f"{combo}_shap_summary_plot_with_percentages.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Return top 2 most important features (by mean absolute SHAP)
    top2_indices = np.argsort(mean_abs_shap)[-2:][::-1]
    top2_features = final_test_data.columns[top2_indices].tolist()
    return top2_features
