#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 11:20:00 2020

@author: shihaoran

this script follows the main flow in BERT_embeddings.py
but ignores the source tags in the captions
"""

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import logging
logging.basicConfig(level=logging.INFO)

import json
import tqdm
import random
import numpy as np
from scipy.spatial.distance import cosine

import tw_funcs as tw

#%%
# load BERT tokenzier
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#%%
# load data set
with open('dataset/merged_dictionary.json') as json_file:
    merged = json.load(json_file)

num_captions = 3150
merged[:] = merged[:num_captions]

#%%
# extract original sentences
X = [i['cleaned_sentence'] for i in merged]
#%%
# add marks and tokenize
X_tokenized = [tokenizer.tokenize('[CLS] ' + text + ' [SEP]') for text in X]
#%%
# convert tokens into indices
X_indexed = [tokenizer.convert_tokens_to_ids(tokens) for tokens in X_tokenized]
#%%
# load BERT model
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()
#%%
# pbar = tqdm.tqdm(total=len(X_indexed))

# X_embeddings = []
# for indices, weight in X_indexed:
#%%

rand_idx = random.randint(0, len(X)-1)
# rand_idx = 50

indices = X_indexed[rand_idx]
text_tokens = X_tokenized[rand_idx]
original_sentence = merged[rand_idx]['original_sentence']
 
# create token and segment tensors
segment_ids = [1] * len(indices)
token_tensor = torch.tensor([indices])
segment_tensor = torch.tensor([segment_ids])
# get encoded layers
with torch.no_grad():
    encoded_layers, _ = model(token_tensor, segment_tensor)
# stack all layers
token_embeddings = torch.stack(encoded_layers, dim=0)

# remove batch dimension since we only have one sample
token_embeddings = torch.squeeze(token_embeddings, dim=1)
# swap dimensions to put the token index at the first dimension, 
# not the layer index
token_embeddings = token_embeddings.permute(1, 0, 2)

# get token embeddings by summing up the last 4 layers
token_vecs_sum = []
for token in token_embeddings:
    sum_vec = torch.sum(token[-4:], dim=0)
    token_vecs_sum.append(sum_vec)

# get sentence embeddings
# which the second to last hidden layer
sentence_vecs = encoded_layers[11][0]
# average
sentence_embedding = torch.mean(sentence_vecs, dim=0)

# calculate similarity of each BERT token
simi_lst = [1-cosine(token_vec, sentence_embedding) for token_vec in token_vecs_sum]

# merge BERT tokens to word tokens
# i.e. remove ## symbols
merged_tokens, merged_simi, merged_embeddings = tw.merged_BERT_tokens(text_tokens, token_vecs_sum, sentence_embedding)

# construct a sample object

sample_obj = {
    'original_sentence': original_sentence,
    'BERT_tokens': text_tokens,
    'BERT_simi': simi_lst,
    'merged_tokens': merged_tokens,
    'merged_simi': merged_simi,
    'keywords': 'None',
    }

tokens_for_ordering = [(i,
                        sample_obj['merged_simi'][i],
                        sample_obj['merged_tokens'][i]) for i in range(len(sample_obj['merged_simi']))]

# order the tuples by descending similarity
tokens_for_ordering.sort(key=lambda x: x[1], reverse=True)
# keep only the top half, excluding the starting and closing tokens
candidate_tokens = tokens_for_ordering[:len(tokens_for_ordering)//2-1]

# sort the candidates back to their original order
candidate_tokens.sort(key=lambda x: x[0])


# get indices
keyword_indices = [candidate[0] for candidate in candidate_tokens]
# get subsets
keyword_indice_subsets = tw.get_subsets(keyword_indices)
# calculate the merged similarity for each subset
subset_simis = [(tw.cal_merged_simi([merged_embeddings[i] for i in subset], sentence_embedding), subset) for subset in keyword_indice_subsets]

# sort
subset_simis.sort(key=lambda x: x[0], reverse=True)
# top3 subsets
top3candidates = [i[1] for i in subset_simis[:3]]

# keyphrases
keyphrases = [" ".join([merged_tokens[i].capitalize() for i in j]) for j in top3candidates]

# get the keywords
keywords = ", ".join(keyphrases)

sample_obj['keywords'] = keywords

tw.plot_simi(sample_obj)
#%%
# X_embeddings.append([token_vecs_sum, sentence_embedding, weight])
    
#     pbar.update(1)

# pbar.close()

# #%%
# # save
# with open('dataset/merged_BERT_embeddings.json', 'w') as json_file:
#     json.dump(X_embeddings, json_file)