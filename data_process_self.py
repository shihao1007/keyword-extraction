#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 10:42:22 2020

@author: shihao
"""


import tw_funcs as tw

#%%
# globals

# minimal number of words to consider as a valid sentence
min_words = 3

#%%
# open source short text
data_dir = 'open_source_data'
X_short = tw.load_data_self_labeling(data_dir)
X_short = tw.clean_set(X_short, min_words)
#%%
# hashtags labeled
data_dir = 'dataset/trnTweet'
X_hash_train = tw.load_data_self_labeling(data_dir)
X_hash_train = tw.clean_set(X_hash_train, min_words)
#%%
data_dir = 'dataset/testTweet'
X_hash_test = tw.load_data_self_labeling(data_dir)
X_hash_test = tw.clean_set(X_hash_test, min_words)
#%%
# dataminr captions
data_dir = 'dataset/Houston_Hurricane_Harvey_2017-08-23__2017-09-01.json'
X_harvey = tw.load_captions(data_dir)
X_harvey = tw.clean_set(X_harvey, min_words)
data_dir = 'dataset/Syria_Late_October_2019_2019-10-24__2019-10-31.json'
X_syria = tw.load_captions(data_dir)
X_syria = tw.clean_set(X_syria, min_words)
#%%
# merge
X_cap = X_harvey + X_syria
X_hash = X_hash_train + X_hash_test
datasets = [X_cap, X_hash, X_short]
merged = tw.weight_balance(datasets)
#%%
tw.save_as_json(merged, 'dataset/merged_dictionary.json')