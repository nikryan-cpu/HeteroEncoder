import pandas as pd
import numpy as np
from scipy.stats import rankdata

def get_exprank_df(df, sf_columns_list, sf_sigmas_list, higher_is_better=True):
    df = df.copy(deep=True)
    if higher_is_better:
        sign = -1
    else:
        sign = 1
    for column in sf_columns_list:
        df[f"{column} rank"] = rankdata(sign*df[column])
    
    sf_num = len(sf_columns_list)
    sf_rank_columns_name = [f"{column} rank" for column in sf_columns_list]
    df["ECR"] = df[sf_rank_columns_name].apply(lambda x: sum([np.exp(-x[i] / sf_sigmas_list[i]) / sf_sigmas_list[i] for i in range(sf_num)]), axis=1)
    return df.sort_values(by="ECR", ascending=False).reset_index(drop=True)
