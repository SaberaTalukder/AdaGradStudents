import numpy as np
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

"""
Pre-process the features
"""


def preprocess(df):
    bid_cols = ['bid1','bid2', 'bid3', 'bid4', 'bid5']
    bid_vol_cols = ['bid1vol', 'bid2vol', 'bid3vol', 'bid4vol', 'bid5vol']
    ask_cols = ['ask1', 'ask2', 'ask3', 'ask4', 'ask5',]
    ask_vol_cols = ['ask1vol','ask2vol', 'ask3vol', 'ask4vol', 'ask5vol']

    group_cols = {"bid_cols":bid_cols,"bid_vol_cols":bid_vol_cols,"ask_cols":ask_cols,"ask_vol_cols":ask_vol_cols}
    for col in bid_cols:
        df[f"{col}_sub"] = (df[col] - df["last_price"])/df["last_price"]
    for col in ask_cols:
        df[f"{col}_sub"] = (df[col] - df["last_price"])/df["last_price"]
    for i in range(5):
        df[f"{i}_imbal"] = (df[bid_vol_cols[i]]-df[ask_vol_cols[i]])/(df[bid_vol_cols[i]]+df[ask_vol_cols[i]])

    df["mid_sub"] = (df["mid"] - df["last_price"])/df["last_price"]

    df["interest_imbal"] = df["d_open_interest"]/df["transacted_qty"]
    df["opened_rat"] = df["opened_position_qty "]/df["transacted_qty"]
    df["closed_rat"] = df["closed_position_qty"]/df["transacted_qty"]

    df = df.drop(columns=bid_cols)
    df = df.drop(columns=ask_cols)
    return df
