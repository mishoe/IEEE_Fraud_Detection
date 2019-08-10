import pandas as pd
import numpy as np
from preprocessor import CatFromTextEncoder, NumericImputer
from scipy import sparse
from seaborn import distplot

train_ident = pd.read_csv('data/train_identity.csv')
train_trans = pd.read_csv('data/train_transaction.csv')
train_df = pd.merge(train_trans, train_ident,on ='TransactionID', how='outer')
df=train_df.drop(columns='isFraud')
numeric_cols = [col for col in df.columns if df[col].dtype!='O']
cat_cols = [col for col in df.columns if df[col].dtype=='O']


## engineer features on the categorical columns
df[cat_cols]=df[cat_cols].fillna("None")
df['matching_domain_flag']=[1 if p==r!="None" else 0 for p,r in df[['P_emaildomain','R_emaildomain']].values]
df['missing_domain_flag']=[1 if p==r=="None" else 0 for p,r in df[['P_emaildomain','R_emaildomain']].values]


num_enc = NumericImputer()
num_enc.fit(df[numeric_cols])
num_data = num_enc.transform(df[numeric_cols])
cat_enc = CatFromTextEncoder()
cat_enc.fit(df[cat_cols])
cat_data = cat_enc.transform(df[cat_cols])

features = sparse.hstack((sparse.coo_matrix(num_data),cat_data))
features = pd.DataFrame(features.todense(),columns=numeric_cols+cat_enc.feature_names)

features.to_pickle('data/features.p')

