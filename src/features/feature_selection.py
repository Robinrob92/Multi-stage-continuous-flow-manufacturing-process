import pandas as pd
from typing import Callable
from sklearn.feature_selection import SelectKBest, f_regression

# Read the pickle file
df = pd.read_pickle('../../data/interim/data_processed.pkl')

# Print column names
for col in df.columns:
    print(col)

# Display the head of the target column
print(df['Stage1.Output.Measurement0.U.Actual'].head())

# Modified function
def select_best_features(df: pd.DataFrame, target_column: str, score_func: Callable = f_regression, k: int = 10) -> pd.DataFrame:
    """
    Select the k best features based on the given score function, excluding columns with "Stage1.Output.Measurement" in their names.

    :param df: The dataframe containing the features and target variable
    :param target_column: The name of the target column in the dataframe
    :param score_func: The scoring function to rank the features (default is f_regression)
    :param k: The number of best features to select (default is 10)
    :return: A dataframe containing only the k best features
    """

    # Filter out columns with "Stage1.Output.Measurement" in their names
    feature_columns = [col for col in df.columns if "Stage1.Output.Measurement" not in col and col != target_column]

    # Separate the features and the target variable
    X = df[feature_columns]
    y = df[target_column]

    # Select the k best features based on the score function
    selector = SelectKBest(score_func=score_func, k=k)
    selector.fit(X, y)

    # Get the indices of the k best features
    best_feature_indices = selector.get_support(indices=True)

    # Create a new dataframe containing the k best features
    best_features_df = df.iloc[:, best_feature_indices]

    return best_features_df

# Call the function with the modified code
result = select_best_features(df=df, target_column='Stage1.Output.Measurement0.U.Actual', k=10)

result.to_pickle("../../data/processed/best_features.pkl")
