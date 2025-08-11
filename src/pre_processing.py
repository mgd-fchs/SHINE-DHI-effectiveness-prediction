import numpy as np

def find_highly_correlated_features(dataframes, threshold=0.8, target_var='responsive'):
    """
    Identifies pairs of highly correlated features in each dataframe.
    :param dataframes: dict of {name: dataframe}
    :param threshold: correlation threshold to consider as "high"
    :return: dict of {name: list of correlated feature pairs}
    """
    correlated_features = {}
    for name, df in dataframes.items():
        # Exclude COMMON_VARS from the correlation computation
        columns_to_correlate = [col for col in df.columns if col != target_var and col !='id']
        
        # Compute correlation matrix only for selected columns
        corr_matrix = df[columns_to_correlate].corr().abs()
        
        # Select the upper triangle of the correlation matrix
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find pairs of features with correlation above the threshold
        correlated_pairs = [
            (col, idx, upper_triangle.loc[idx, col])
            for col in upper_triangle.columns
            for idx in upper_triangle.index
            if upper_triangle.loc[idx, col] > threshold
        ]
        
        # Store results for the current dataframe
        correlated_features[name] = correlated_pairs

    return correlated_features