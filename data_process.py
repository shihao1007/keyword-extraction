#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 10:59:58 2020

@author: shihao
"""


import tw_funcs as tw
import numpy as np
import pandas as pd

from gensim.models import KeyedVectors
#%%
# helper functions
def get_embeddings(word_list):
    """
    map all the words in the original data set into their embeddings
    
    Parameters
    ----------
        word_list:
            a list of tokens for the sentence/keyphrase
            
    Returns
    -------
        embedding_list:
            a list of embedding vectors of the tokens
    """
    
    # check if the pretrained model exists
    try:
        wordEmbeddings
    except (AttributeError):
        print("Word Embedding Model is Not Defined!")
    
    # if the current word exists in the embedding model
    embedding_list = []
    for word in word_list:
        if word in wordEmbeddings:
            embedding_list.append(wordEmbeddings[word])
    return embedding_list
#%%
# load data sets
train_data_dir = 'dataset/trnTweet'
test_data_dir = 'dataset/testTweet'

X_train, y_train = tw.load_data(train_data_dir)
X_test, y_test = tw.load_data(test_data_dir)

#%%
# load embedding model
wordEmbeddings = KeyedVectors.load_word2vec_format('/home/shihao/Data/GoogleNews-vectors-negative300.bin', binary=True)
#%%
# initialize dataframe
df_test = pd.DataFrame()
df_test['X'] = X_test
df_test['y'] = y_test

df_train = pd.DataFrame()
df_train['X'] = X_train
df_train['y'] = y_train
#%%
# get embeddings
df_test['X_embed'] = df_test.X.apply(get_embeddings)

df_train['X_embed'] = df_train.X.apply(get_embeddings)

#%%
# clean dataframe remove all empty embeddings in the labels
# replace empty list with nan
df_test.X_embed = df_test.X_embed.apply(lambda x: np.nan if len(x)==0 else x)
df_test.y = df_test.y.apply(lambda x: np.nan if len(x)==0 else x)

df_train.X_embed = df_train.X_embed.apply(lambda x: np.nan if len(x)==0 else x)
df_train.y = df_train.y.apply(lambda x: np.nan if len(x)==0 else x)
#%%
# drop
df_test.dropna(inplace=True)
df_train.dropna(inplace=True)
#%%
# save
X_test_embeddings = df_test.X_embed.to_numpy()
np.save('dataset/X_test.npy', X_test_embeddings)

X_train_embeddings = df_train.X_embed.to_numpy()
np.save('dataset/X_train.npy', X_train_embeddings)
#%%
y_test = df_test.y.to_numpy()
np.save('dataset/y_test_binary.npy', y_test)

y_train = df_train.y.to_numpy()
np.save('dataset/y_train_binary.npy', y_train)
