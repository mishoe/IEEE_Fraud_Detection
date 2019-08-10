import pandas as pd
import numpy as np
from preprocessor import CatFromTextEncoder, NumericImputer
from scipy import sparse

train_ident = pd.read_csv('data/train_identity.csv')
train_trans = pd.read_csv('data/train_transaction.csv')
train_df = pd.merge(train_trans, train_ident,on ='TransactionID', how='outer')
train_df=train_df.drop(columns='isFraud')

numeric_cols = [col for col in train_df.columns if train_df[col].dtype!='O']
cat_cols = [col for col in train_df.columns if train_df[col].dtype=='O']

num_enc = NumericImputer()
num_enc.fit(train_df[numeric_cols])
num_data = num_enc.transform(train_df[numeric_cols])
cat_enc = CatFromTextEncoder()
cat_enc.fit(train_df[cat_cols])
cat_data = cat_enc.transform(train_df[cat_cols])

features = sparse.hstack((sparse.coo_matrix(num_data),cat_data))
features = pd.DataFrame(features.todense(),columns=numeric_cols+cat_enc.feature_names)

features.to_pickle('data/features.p')
