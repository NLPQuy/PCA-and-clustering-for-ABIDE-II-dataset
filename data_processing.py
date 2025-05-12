import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold


def load_data(file_path="ABIDE2(updated).csv"):
    """
    Load the ABIDE dataset and provide basic information
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    return df


def analyze_basic_stats(df):
    """
    Analyze basic statistics of the dataset
    """
    print("\n=== Basic Dataset Statistics ===")
    # Get basic statistics
    print("\nColumns with unique values:")
    for col in ['site', 'group', 'age']:
        if col in df.columns:
            print(f"{col}: {df[col].unique()}")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("\nMissing values by column:")
        print(missing[missing > 0])
    else:
        print("\nNo missing values found in the dataset.")
    
    # Basic statistics for numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    print("\nNumerical columns statistics:")
    stats_df = df[numerical_cols].describe().T
    stats_df['skewness'] = df[numerical_cols].skew()
    stats_df['kurtosis'] = df[numerical_cols].kurtosis()
    print(stats_df[['count', 'mean', 'std', 'min', 'max', 'skewness', 'kurtosis']].head())
    
    return stats_df


def check_distribution(df, sample_columns=None, n_cols=5):
    """
    Check the distribution of data in selected features
    """
    print("\n=== Distribution Analysis ===")
    
    if sample_columns is None:
        # Get feature columns (excluding metadata)
        feature_cols = df.columns[df.columns.str.contains('fs')].tolist()
        # Select a subset of columns for visualization
        if len(feature_cols) > n_cols:
            sample_columns = np.random.choice(feature_cols, n_cols, replace=False)
        else:
            sample_columns = feature_cols
    
    # Shapiro-Wilk test for normality
    print("\nShapiro-Wilk test for normality (sample of features):")
    for col in sample_columns:
        # Take a sample if there are too many rows
        sample = df[col].dropna()
        if len(sample) > 5000:
            sample = sample.sample(5000)
        
        shapiro_test = stats.shapiro(sample)
        print(f"{col}: W={shapiro_test.statistic:.4f}, p-value={shapiro_test.pvalue:.4e}")
        if shapiro_test.pvalue < 0.05:
            print("  - Not normally distributed")
        else:
            print("  - Normally distributed")
    
    # Plot histograms for sample columns
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(sample_columns[:min(n_cols, len(sample_columns))]):
        plt.subplot(n_cols, 2, 2*i+1)
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        
        plt.subplot(n_cols, 2, 2*i+2)
        stats.probplot(df[col].dropna(), plot=plt)
        plt.title(f"Q-Q Plot of {col}")
    
    plt.tight_layout()
    plt.savefig('plots/distribution_analysis.png')
    plt.close()
    
    return sample_columns


def analyze_group_differences(df):
    """
    Analyze differences between groups (if available)
    """
    if 'group' not in df.columns:
        print("No 'group' column found in the dataset.")
        return
    
    print("\n=== Group Difference Analysis ===")
    groups = df['group'].unique()
    print(f"Groups present in data: {groups}")
    
    # Number of samples per group
    group_counts = df['group'].value_counts()
    print("\nSamples per group:")
    print(group_counts)
    
    # Select a few brain features for comparison
    feature_cols = df.columns[df.columns.str.contains('fs')].tolist()
    selected_features = np.random.choice(feature_cols, 5, replace=False)
    
    # Check for statistical differences between groups
    print("\nMann-Whitney U test (non-parametric) between groups:")
    for feature in selected_features:
        group_data = [df[df['group'] == group][feature].dropna().values for group in groups]
        
        # Skip if any group has no data
        if any(len(gd) == 0 for gd in group_data):
            continue
            
        # Perform Mann-Whitney U test
        u_stat, p_value = stats.mannwhitneyu(group_data[0], group_data[1])
        print(f"{feature}: U={u_stat:.2f}, p-value={p_value:.4e}")
        if p_value < 0.05:
            print(f"  - Significant difference between groups")
        else:
            print(f"  - No significant difference between groups")
    
    # Visualize differences for selected features
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(selected_features):
        plt.subplot(3, 2, i+1)
        sns.boxplot(x='group', y=feature, data=df)
        plt.title(f"{feature} by Group")
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('plots/group_differences.png')
    plt.close()


def analyze_feature_correlations(df, n_features=20):
    """
    Analyze correlations between features
    """
    print("\n=== Feature Correlation Analysis ===")
    
    # Get feature columns (excluding metadata)
    feature_cols = df.columns[df.columns.str.contains('fs')].tolist()
    
    # If too many features, select a subset
    if len(feature_cols) > n_features:
        feature_cols = np.random.choice(feature_cols, n_features, replace=False)
    
    # Calculate correlation matrix
    corr_matrix = df[feature_cols].corr()
    
    # Get the most correlated pairs
    corr_pairs = []
    for i in range(len(feature_cols)):
        for j in range(i+1, len(feature_cols)):
            corr_pairs.append((feature_cols[i], feature_cols[j], corr_matrix.iloc[i, j]))
    
    # Sort by absolute correlation value
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    print("\nTop 10 most correlated feature pairs:")
    for x, y, corr in corr_pairs[:10]:
        print(f"{x} - {y}: {corr:.4f}")
    
    # Visualize correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('plots/correlation_matrix.png')
    plt.close()
    
    return corr_matrix, corr_pairs


def get_optimal_preprocessing(df):
    """
    Determine optimal preprocessing methods based on data characteristics
    """
    print("\n=== Optimal Preprocessing Determination ===")
    
    # Get feature columns (excluding metadata)
    feature_cols = df.columns[df.columns.str.contains('fs')].tolist()
    
    # Calculate skewness for all features
    skewness = df[feature_cols].skew()
    highly_skewed = skewness[abs(skewness) > 1].index.tolist()
    moderately_skewed = skewness[(abs(skewness) > 0.5) & (abs(skewness) <= 1)].index.tolist()
    
    print(f"\nHighly skewed features (|skew| > 1): {len(highly_skewed)}")
    print(f"Moderately skewed features (0.5 < |skew| <= 1): {len(moderately_skewed)}")
    
    # Check for outliers
    has_outliers = False
    for col in feature_cols[:min(20, len(feature_cols))]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        if len(outliers) > 0:
            has_outliers = True
            break
    
    print(f"\nOutliers detected: {has_outliers}")
    
    # Recommend preprocessing techniques
    print("\nRecommended preprocessing techniques:")
    
    if len(highly_skewed) > 0:
        print("- Power transformation (Yeo-Johnson) for highly skewed features")
    
    if has_outliers:
        print("- Robust scaling to handle outliers")
    else:
        print("- Standard scaling for normally distributed features")
    
    print("- Feature selection to remove highly correlated features")
    print("- Dimensionality reduction (PCA) to handle high feature count")
    
    return {
        'highly_skewed': highly_skewed,
        'moderately_skewed': moderately_skewed,
        'has_outliers': has_outliers
    }


def preprocess_data(df, preprocess_params=None):
    """
    Apply optimal preprocessing methods to the dataset
    """
    print("\n=== Applying Preprocessing ===")
    
    # Make a copy of the original dataframe
    processed_df = df.copy()
    
    # Separate metadata and features
    metadata_cols = [col for col in df.columns if not col.startswith('fs')]
    feature_cols = [col for col in df.columns if col.startswith('fs')]
    
    print(f"Metadata columns: {len(metadata_cols)}")
    print(f"Feature columns: {len(feature_cols)}")
    
    # Extract features
    X = processed_df[feature_cols].copy()
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=feature_cols)
    
    # If preprocess_params not provided, analyze data to determine optimal preprocessing
    if preprocess_params is None:
        preprocess_params = get_optimal_preprocessing(df)
    
    # Apply transformations based on data characteristics
    
    # 1. Apply power transform to highly skewed features
    if len(preprocess_params.get('highly_skewed', [])) > 0:
        print("\nApplying power transformation to highly skewed features...")
        pt = PowerTransformer(method='yeo-johnson')
        highly_skewed = preprocess_params['highly_skewed']
        X_skewed = X[highly_skewed]
        X_skewed_transformed = pt.fit_transform(X_skewed)
        X[highly_skewed] = X_skewed_transformed
    
    # 2. Apply scaling
    if preprocess_params.get('has_outliers', False):
        print("Applying robust scaling to handle outliers...")
        scaler = RobustScaler()
    else:
        print("Applying standard scaling...")
        scaler = StandardScaler()
    
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=feature_cols)
    
    # 3. Remove low variance features
    print("Removing low variance features...")
    selector = VarianceThreshold(threshold=0.01)
    X_var_filtered = selector.fit_transform(X)
    selected_features = [feature_cols[i] for i in range(len(feature_cols)) if selector.get_support()[i]]
    X = pd.DataFrame(X_var_filtered, columns=selected_features)
    
    print(f"Features after variance filtering: {X.shape[1]}")
    
    # Optionally: Apply PCA for dimensionality reduction
    
    # Combine metadata with processed features
    for col in metadata_cols:
        X[col] = processed_df[col].values
    
    print(f"Final processed dataset shape: {X.shape}")
    return X


def main():
    """
    Main function to run all analyses
    """
    # Create plots directory if it doesn't exist
    import os
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # Load data
    df = load_data()
    
    # Analyze basic statistics
    stats_df = analyze_basic_stats(df)
    
    # Check distribution
    sample_columns = check_distribution(df)
    
    # Analyze group differences
    analyze_group_differences(df)
    
    # Analyze feature correlations
    corr_matrix, corr_pairs = analyze_feature_correlations(df)
    
    # Determine optimal preprocessing
    preprocess_params = get_optimal_preprocessing(df)
    
    # Apply preprocessing
    processed_df = preprocess_data(df, preprocess_params)
    
    # Save processed data
    processed_df.to_csv('ABIDE2_processed.csv', index=False)
    print("\nProcessed data saved to 'ABIDE2_processed.csv'")


if __name__ == "__main__":
    main() 