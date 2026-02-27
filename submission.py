import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

def removing_values (data, target):
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
    
    return data, target

def fft_cols(data, name_col):
    
    new_cols_fft =[]
    
    for l in range(len(data[name_col])):
        new_cols_fft.append(abs(scipy.fft.fft(data[name_col][l], n = 50)))
    
    expanded_cols = pd.DataFrame(new_cols_fft, columns=[f'{name_col}_{i}' for i in range(len(new_cols_fft[0]))])

    data = data.drop(columns=[name_col])

    data_final = pd.concat([data.reset_index(drop=True), expanded_cols.reset_index(drop=True)], axis=1)

    return data_final


cat_to_int = ColumnTransformer([
    ("OneHotEncoder", OneHotEncoder(), ["gender", "domain"])
])

def xgb_cross(X_df,y):
    mod = GradientBoostingRegressor(max_depth = 3, min_child_weight=5, learning_rate = 0.22588673527216052, n_estimators = 415)

    return cross_val_score(mod, X_df, pd.Series(y), cv=5, scoring='neg_mean_absolute_error').mean()

model = Pipeline([
    ("Delete_columns",removing_values())
    ("OneHotEncoder", cat_to_int),
    ("Fourier for ppg ", fft_cols(name_col="ppg"))
    ("Fourier for ecg ", fft_cols(name_col="ecg"))
    ("Regressor", xgb_cross()),
])