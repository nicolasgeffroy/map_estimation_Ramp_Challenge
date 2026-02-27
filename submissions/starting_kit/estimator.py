import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import GradientBoostingRegressor

""" def removing_values (data, target):
    data = data.drop(np.where(target == -1)[0].tolist())
    target = target[target != -1]
    data = data.drop(["n_seg", "subject", "id"], axis="columns")
    
    na_index = data[data.isna().any(axis=1)].index
    na_bool = []

    for i in data.index:
        if i not in list(na_index):
            na_bool.append(True)
        else:
            na_bool.append(False)
    data = data[na_bool]
    target = target[na_bool]
    
    return data, target """

def fft(x, n=None):
    if n is None:
        n = len(x)
    elif n < len(x):
        x = x[:n]  # Crop the input if n is smaller
    elif n > len(x):
        x = x + [0] * (n - len(x))  # Pad with zeros if n is larger
    
    N = len(x)
    if N <= 1:
        return x
    
    if N % 2 > 0:
        np.append(x,0)  # Ensure the length is even for splitting
        N += 1
    
    even = fft(x[0::2], N // 2)
    odd = fft(x[1::2], N // 2)
    
    T = [np.exp(-2j * np.pi * k / N) * odd[k % len(odd)] for k in range(N // 2)]
    return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]


def fft_cols(data, name_col):
    
    new_cols_fft =[]
    
    for l in range(len(data[name_col])):
        res_fft = fft(data[name_col][l], n = 15)
        new_cols_fft.append(abs(np.array(res_fft)))
    
    expanded_cols = pd.DataFrame(new_cols_fft, columns=[f'{name_col}_{i}' for i in range(len(new_cols_fft[0]))])

    data = data.drop(columns=[name_col])

    data_final = pd.concat([data.reset_index(drop=True), expanded_cols.reset_index(drop=True)], axis=1)
    
    data_final = data_final.rename({0: "0_dom", 1: "1_dom"}, axis='columns')

    return data_final

cat_to_int = make_column_transformer(
    ("OneHotEncoder", OneHotEncoder(handle_unknown = "ignore"), ["gender", "domain"]),
    ('passthrough', ['height', 'weight', 'bmi', 0, 1])
)

ftt_ppg_func = FunctionTransformer(
    lambda X_df: fft_cols(X_df, name_col = "ppg")
)
ftt_ecg_func = FunctionTransformer(
    lambda X_df: fft_cols(X_df, name_col = "ecg")
)

cols = ['height', 'weight', 'bmi', "0_dom", "1_dom", 'ppg_0', 'ppg_1', 'ppg_2', 'ppg_3', 'ppg_4', 'ppg_5', 'ppg_6', 'ppg_7', 'ppg_8', 'ppg_9', 'ppg_10', 'ppg_11', 'ppg_12', 'ppg_13', 'ppg_14', 'ppg_15', 'ecg_0',  'ecg_1',  'ecg_2',  'ecg_3',  'ecg_4',  'ecg_5',  'ecg_6', 'ecg_7', 'ecg_8', 'ecg_9', 'ecg_10', 'ecg_11', 'ecg_12', 'ecg_13', 'ecg_14', 'ecg_15']

transformer_ppg_ecg = make_column_transformer(
    (ftt_ppg_func, ["ppg"]),
    (ftt_ecg_func, ["ecg"]),
    ('passthrough', cols)
)

model = make_pipeline(
    transformer_ppg_ecg,
    ("OneHotEncoder", cat_to_int),
    SimpleImputer(strategy=callable),
    ("Regressor", GradientBoostingRegressor(learning_rate = 0.4533395920407246, n_estimators = 318, 
                                            max_depth = 5, min_samples_split = 17, min_samples_leaf = 9))
)

def get_estimator():
    return model