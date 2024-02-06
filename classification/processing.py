import os

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATAFRAME_FILE_PATH = os.path.join(
    BASE_DIR,
    "data",
    "classification_dataframe",
    "dataset.csv",
)

DATE_COL: str = "Date"
TARGET_COL: str = "ignition"


class DataProcessor:
    """
    Data processing pipeline
    """

    def __init__(self) -> None:
        df = self.load_data()
        self.df = self.add_date_features(df)

    @st.cache_data
    @staticmethod
    def load_data(_) -> pd.DataFrame:
        df = pd.read_csv(DATAFRAME_FILE_PATH, parse_dates=[DATE_COL], index_col=0)
        return df

    @st.cache_data
    def add_date_features(_self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform feature engineering on Date column
        """
        df_copy = df.copy()

        df_copy["day"] = df_copy[DATE_COL].dt.day
        df_copy["day_sin"] = np.sin(2 * np.pi * df_copy["day"] / 12)

        df_copy["month"] = df_copy[DATE_COL].dt.month
        df_copy["month_sin"] = np.sin(2 * np.pi * df_copy["month"] / 12)

        df_copy["days_since_reference"] = (
            df_copy[DATE_COL] - df_copy[DATE_COL].min()
        ).dt.days

        return df_copy.drop(columns=[DATE_COL])

    @st.cache_data
    def split_data(
        _self,
        test_size: float = 0.25,
        random_state: int = 0,
        shuffle: bool = True,
        stratify_flag: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test
        """
        X, y = _self.df.drop(columns=TARGET_COL), _self.df[TARGET_COL]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=y if (stratify_flag and shuffle) else None,
        )

        return X_train, X_test, y_train, y_test
