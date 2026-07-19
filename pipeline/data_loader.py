import pandas as pd
import re

def clean_column_names(df):
    df = df.copy()
    df.columns = [re.sub(r'[\[\]<]', '', col) for col in df.columns]
    return df

def load_and_prepare(df, config):
    df = df.drop(columns=config["id_cols"] + config["drop_cols"])
    X = df[config["numeric_features"] + config["categorical_features"]]
    y = df[config["target_col"]]
    if config["categorical_features"]:
        X = pd.get_dummies(X, columns=config["categorical_features"], drop_first=True)
    X = clean_column_names(X)
    return X, y