import numpy as np
import pandas as pd

from features import number_feature, category_feature, sequence_feature
from transformers.column import (
    DoNone,StandardScaler, CategoryEncoder, SequenceEncoder)

from utils import dataset
from model import deepfm,DIN,AttentionGroup
from utils.functions import fit, predict, create_dataloader_fn

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import time
from utils.progressbar import ProgressBar

from sklearn.preprocessing import LabelEncoder


start_time = time.time()
# pbar = ProgressBar(n_total=30,desc='read-train')
# train = []
# train_reader = pd.read_csv('../user_data/data/train_tmp.csv',chunksize=100000)
# for i,ck in enumerate(train_reader):
#     train.append(ck)
#     pbar(step = i)
# train_df = pd.concat(train, ignore_index=True)

# pbar = ProgressBar(n_total=30,desc='read-valid')
# test = []
# valid_reader = pd.read_csv('../user_data/data/test_tmp.csv',chunksize=100000)
# for i,ck in enumerate(valid_reader):
#     test.append(ck)
#     pbar(step = i)
# valid_df = pd.concat(test, ignore_index=True)

train_df = pd.read_csv('../user_data/data/train_tmp.csv',nrows=100)
valid_df = pd.read_csv('../user_data/data/test_tmp.csv',nrows=10)
print('读取消耗时间：{}'.format(time.time()-start_time))

start_time_ = time.time()
train_df["previous_feed_embedding"] = train_df["previous_feed_embedding"].apply(lambda x: x.replace("[", ""))
train_df["previous_feed_embedding"] = train_df["previous_feed_embedding"].apply(lambda x: x.replace("]", ""))
train_df["previous_feed_embedding"] = train_df["previous_feed_embedding"].apply(lambda x: x.replace("\n", ""))
# train_df["previous_feed_embedding"] = train_df["previous_feed_embedding"].apply(lambda x: x.split(" "))
train_df["previous_feed_embedding"] = train_df["previous_feed_embedding"].apply(lambda x: np.array([np.float32(i) for i in x.split(" ") if i != '']))

train_df["current_feed_embedding"] = train_df["current_feed_embedding"].apply(lambda x: x.replace("[", ""))
train_df["current_feed_embedding"] = train_df["current_feed_embedding"].apply(lambda x: x.replace("]", ""))
train_df["current_feed_embedding"] = train_df["current_feed_embedding"].apply(lambda x: x.replace("\n", ""))
# train_df["current_feed_embedding"] = train_df["current_feed_embedding"].apply(lambda x: x.split(" "))
train_df["current_feed_embedding"] = train_df["current_feed_embedding"].apply(lambda x: np.array([np.float32(i) for i in x.split(" ") if i != '']))

valid_df["previous_feed_embedding"] = valid_df["previous_feed_embedding"].apply(lambda x: x.replace("[", ""))
valid_df["previous_feed_embedding"] = valid_df["previous_feed_embedding"].apply(lambda x: x.replace("]", ""))
valid_df["previous_feed_embedding"] = valid_df["previous_feed_embedding"].apply(lambda x: x.replace("\n", ""))
# valid_df["previous_feed_embedding"] = valid_df["previous_feed_embedding"].apply(lambda x: x.split(" "))
valid_df["previous_feed_embedding"] = valid_df["previous_feed_embedding"].apply(lambda x: np.array([np.float32(i) for i in x.split(" ") if i != '']))

valid_df["current_feed_embedding"] = valid_df["current_feed_embedding"].apply(lambda x: x.replace("[", ""))
valid_df["current_feed_embedding"] = valid_df["current_feed_embedding"].apply(lambda x: x.replace("]", ""))
valid_df["current_feed_embedding"] = valid_df["current_feed_embedding"].apply(lambda x: x.replace("\n", ""))
# valid_df["current_feed_embedding"] = valid_df["current_feed_embedding"].apply(lambda x: x.split(" "))
valid_df["current_feed_embedding"] = valid_df["current_feed_embedding"].apply(lambda x: np.array([np.float32(i) for i in x.split(" ") if i != '']))
print('处理数据数据消耗时间：{}'.format(time.time()-start_time_))
sparse_features = ['userid','feedid','date_','device']
dense_features = ['play','stay']
label_names = ['read_comment','like','click_avatar','forward']
info_features = ['feedid','authorid','bgm_song_id','bgm_singer_id']

number_features = [
    number_feature.Number('previous_feed_embedding', DoNone()),
    number_feature.Number('current_feed_embedding', DoNone()),
]

# number_features = [
#     number_feature.Number('play', StandardScaler()),
#     number_feature.Number('stay', StandardScaler()),
# ]

category_features = [
    category_feature.Category('userid', LabelEncoder()),
    category_feature.Category('feedid', LabelEncoder()),
    # category_feature.Category('date_', CategoryEncoder(min_cnt=1)),
    category_feature.Category('device', LabelEncoder()),
    # category_feature.Category('authorid', CategoryEncoder(min_cnt=1)),
    # category_feature.Category('bgm_song_id', CategoryEncoder(min_cnt=1)),
    # category_feature.Category('bgm_singer_id', CategoryEncoder(min_cnt=1)),
]

# sequence_features = [
#     sequence_feature.Sequence('high_rated_feedids', SequenceEncoder(sep='|', min_cnt=1)),
# ]

sequence_features = []

feed_info = pd.read_csv('../data/wechat_algo_data1/feed_info.csv',low_memory=False)
user_action = pd.read_csv('../data/wechat_algo_data1/user_action.csv',low_memory=False)

feed_info = feed_info.iloc[:feed_info.shape[0]-2]
users = user_action.userid.astype(float).unique()
feeds = feed_info.feedid.astype(float).unique()

features, train_loader,test_loader,valid_loader, = create_dataloader_fn(
        number_features, category_features,sequence_features, 64, train_df, label_names, valid_df,valid_df,4,users,feeds)
print('创建完毕')
model = deepfm.DeepFM(features = features, num_classes = 4, embedding_size = 32, 
                     hidden_layers = (32,16), activation='relu',final_activation='direct_output', 
                     dropout=0.3,use_linear = True)

# din_attention_groups = [
#     AttentionGroup(
#         name='group1',
#         pairs=[{'ad': 'feedid', 'pos_hist': 'previous_feed_embedding'}],
#         hidden_layers=[16, 8], att_dropout=0.1)]
# model = DIN(features = features, attention_groups = din_attention_groups, 
#             num_classes = 4,embedding_size = 8, hidden_layers = (32,16),
#             final_activation='direct_output', dropout=0.1)


print(model)
loss_func = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = 2,eta_min=0)
fit(5, model, loss_func, optimizer,
    train_loader, valid_loader, notebook=True,
    scheduler = scheduler,model_save_path = '../user_data/model/')

predict(features = features,model_path = '../user_data/model/model.pkl',
        test_loader = test_loader)

print('消耗时间：{}'.format(time.time()-start_time))