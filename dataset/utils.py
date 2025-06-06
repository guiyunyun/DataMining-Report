import pandas as pd
import os
from sklearn.utils import shuffle


labels = ['severe', 'moderate', 'not depression']


text_column_names = {
    "dev": "Text data",
    "train": "Text_data",
    "test": "text data"
}

pid_column_names = {
    "dev": "PID",
    "train": "PID",
    "test": "Pid"
}

label_column_names = {
    "dev": "Label",
    "train": "Label",
    "test": "Class labels"
}


def get_data(data_split, use_shuffle=False, without_label=False):
    data_path = os.path.join('data', 'original_dataset', f'{data_split}.tsv')
    df = pd.read_csv(data_path, sep='\t', header=0)

    pid_column = pid_column_names.get(data_split)
    text_column = text_column_names.get(data_split)
    label_column = label_column_names.get(data_split)
    if without_label:
        df = df[[pid_column, text_column]]
        df.columns = ["pid", "text"]
    else:
        df[label_column] = df[label_column].transform(lambda label: labels.index(label))
        df.columns = ["pid", "text", "labels"]
    if use_shuffle:
        return shuffle(df)
    return df


def get_preprocessed_data(data_split, use_shuffle=False):
    data_path = os.path.join('data', 'preprocessed_dataset', f'{data_split}.csv')
    df = pd.read_csv(data_path, header=0, lineterminator='\n')
    if use_shuffle:
        return shuffle(df)
    return df
