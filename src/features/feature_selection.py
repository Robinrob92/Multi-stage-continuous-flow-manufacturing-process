import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from typing import Callable

df=pd.read_pickle('../../data/interim/data.pkl')


def select_best_features(df: pd.DataFrame, target_column: str, score_func: Callable = f_regression, k: int = 10) -> pd.DataFrame:
    """
    Select the k best features based on the given score function.

    :param df: The dataframe containing the features and target variable
    :param target_column: The name of the target column in the dataframe
    :param score_func: The scoring function to rank the features (default is f_regression)
    :param k: The number of best features to select (default is 10)
    :return: A dataframe containing only the k best features and the target column
    """

    # Separate the features and the target variable
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Select the k best features based on the score function
    selector = SelectKBest(score_func=score_func, k=k)
    selector.fit(X, y)

    # Get the indices of the k best features
    best_feature_indices = selector.get_support(indices=True)

    # Create a new dataframe containing the k best features and the target column
    best_features_df = df.iloc[:, best_feature_indices].join(y)

    return best_features_df
