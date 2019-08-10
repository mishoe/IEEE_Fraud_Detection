import xgboost as xgb
import pandas as pd
import numpy as np
import random
from xgb_optimizer import XgboostOptimizer
from plot_methods import plot_information_gain
import pickle

target_col = 'isFraud'
data = pd.read_csv('data/train_transaction.csv')
features = pd.read_pickle('data/features.p')

#
# keep_inds = np.where(data['MinOH4'].isna()==False)[0]
# data = data.iloc[keep_inds,:].reset_index(drop=True)
# features = features.iloc[keep_inds,:].reset_index(drop=True)

# data[target_col] = list(map(lambda x: 1 if x>0 else 0,data['FutureTD']))

train_test_prop = .8
random.seed(0)
train_inds = random.sample(population=range(features.shape[0]),k = int(train_test_prop*features.shape[0]))
test_inds = list(set(range(features.shape[0])) - set(train_inds))


train = xgb.DMatrix(features.iloc[train_inds,:],label = data.loc[train_inds,target_col],feature_names = features.columns)
test = xgb.DMatrix(features.iloc[test_inds,:],label = data.loc[test_inds,target_col],feature_names = features.columns)


train_params = { 'objective': 'binary:logistic',
                 'eval_metric': 'auc',
                 'max_depth' : range(1, 7),
                 'build_past': 3,
                 'early_stopping_rounds': 3,
                 'max_minutes_total':480,
                 'num_rand_samples': 35,
                 }
x = XgboostOptimizer()
x.update_params(train_params)
x.fit_random_search(dtrain=train,evals=[(train, 'Train'), (test, "Test")])
print('done with random search')
x.fit_seq_search(dtrain=train,evals=[(train, 'Train'), (test, "Test")])
model = x.best_model
print(x.best_score)

plot_information_gain(model,25)


pickle.dump(model,open('models/'+target_col+'_Model.p','wb'))



#check the optimal cutoff vals
preds = model.predict(test)
# from utils import calculate_optimal_cutoff
# optim_cuts = calculate_optimal_cutoff([int(val) for val in test.get_label()],preds)


# from sklearn.metrics import classification_report
# print(classification_report([int(val) for val in test.get_label()], [1 if val>=.02 else 0 for val in preds]))
#
#
# true = test.get_label()
# import seaborn as sns
# from matplotlib import pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# sns.distplot(np.log([1+100*preds[i] for i in range(len(preds)) if true[i]==1]))
# sns.distplot(np.log([1+100*preds[i] for i in range(len(preds)) if true[i]==0]))
# labs = ax.get_xticklabels()
# ax.set_xticklabels()
# plt.show()
