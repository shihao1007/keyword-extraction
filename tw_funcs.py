#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 10:59:58 2020

@author: shihao

This is all the helper functions used in TweetWather
"""
# import tensorflow as tf
import torch
import matplotlib.pyplot as plt
import numpy as np
from gensim.models import KeyedVectors
import json
import re
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stops = stopwords.words('english')

from scipy.spatial.distance import cosine

def get_subsets(lst):
    res = [[]]
    for i in lst:
        res += [x+[i] for x in res]
    return res[1:]
    

def plot_graphs(history, metric):
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])
    plt.show()
    
def load_data(data_dir):
    X, y = [], []
    with open(data_dir) as data_file:
        for line in data_file:
            X_curr, y_curr = [i.split() for i in line.split('\t')]
            X.append(X_curr)
            y.append([1 if word in y_curr else 0 for word in X_curr])
    return X, y

def load_data_self_labeling(data_dir):
    X= []
    with open(data_dir) as data_file:
        for line in data_file:
            X_curr = line.split('\t')[0]
            X.append(X_curr)
    return X

def load_captions(data_dir):
    
    X = []
    with open(data_dir) as json_file:
        lst = json.load(json_file)
        for dit in lst:
            alerts = dit['alerts']
            if not alerts:
                continue
            for alert in alerts:
                if 'keywords' in alert:
                    X.append(alert['caption'].split(':')[0])
    return X

def clean_text(sentence):
    
    obj = {'original_sentence': sentence}
    
    # remove html
    sentence = re.sub(r'http?:\/\/.*[\r\n]*', '', sentence)
    tokens = sentence.split()
    table = str.maketrans('', '', string.punctuation)
    stripped = [token.translate(table) for token in tokens]
    
    cleaned_tokens = [i for i in stripped if i != '' and i.lower() not in stops]
    
    obj['tokens'] = cleaned_tokens
    
    return obj

def clean_set(dataset, min_words):
    
    cleaned = list(map(clean_text, dataset))
    cleaned_set = []
    
    for obj in cleaned:
        
        if len(obj['tokens']) >= min_words:
            
            obj['cleaned_sentence'] = " ".join(obj['tokens'])
            cleaned_set.append(obj)
            
    
    return cleaned_set

def weight_balance(datasets):
    max_sample_num = max([len(dataset) for dataset in datasets])
    merged = []
    for dataset in datasets:
        curr_weight = max_sample_num/len(dataset)
        for obj in dataset:
            obj['weight'] = curr_weight
            merged.append(obj)
    return merged

def save_as_json(dataset, data_dir):
    with open(data_dir, 'w') as json_file:
        json.dump(dataset, json_file)


def make_pred(sample, model, padding):
    
    
    if 'wordEmbeddings' not in globals():
        wordEmbeddings = KeyedVectors.load_word2vec_format('/home/shihao/Data/GoogleNews-vectors-negative300.bin', binary=True)

    tokens = sample.split()
    tokens_filtered = [token for token in tokens if token in wordEmbeddings]
    
    token_dit = {idx: token for idx, token in enumerate(tokens_filtered)}
    sample_embeddings = [[wordEmbeddings[word] for word in tokens if word in wordEmbeddings]]
    # sample_embeddings = [[wordEmbeddings[word] for word in tokens if word in wordEmbeddings]]
    
    sample_embeddings = tf.keras.preprocessing.sequence.pad_sequences(
        sample_embeddings, maxlen=padding, dtype='float32', padding='post', truncating='post',
        value=0.0
    )
    
    pred = list(model.predict(sample_embeddings).squeeze())
    
    print(' '.join([token_dit[i] if pred[i] > 0.05 else '' for i in range(padding)]))
    
def plot_simi(sample_obj, figheight=3):
    
    # BERT related info
    BERT_similarity = np.array(sample_obj['BERT_simi'])
    BERT_tokens = sample_obj['BERT_tokens']
    
    merged_similarity = np.array(sample_obj['merged_simi'])
    merged_tokens = sample_obj['merged_tokens']

    # get figure width according to the number of 
    # tokens in BERT list
    figwidth = len(BERT_tokens) * 6
    
    BERT_token_lens = np.array([len(t) for t in BERT_tokens])
    BERT_token_lens_cum = BERT_token_lens.cumsum()
    BERT_norm = BERT_similarity/np.max(BERT_similarity)
    BERT_colors = plt.get_cmap('GnBu')(BERT_norm)
    
    
    # merged info
    merged_token_lens = np.array([len(t) for t in merged_tokens], dtype=np.float)
    merged_token_lens_cum = merged_token_lens.cumsum()
    merged_norm = merged_similarity/np.max(merged_similarity)
    merged_colors = plt.get_cmap('GnBu')(merged_norm)
    
    # rescale ratio
    ratio = BERT_token_lens_cum[-1]/merged_token_lens_cum[-1]
    
    merged_token_lens_cum *= ratio
    
    # figure configurations
    fig, ax = plt.subplots(figsize=(figwidth, figheight))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_ylim(figheight)
    
    # plot BERT tokens
    BERT_bar_height = figheight/4+0.2
    
    for i, (token, color) in enumerate(zip(BERT_tokens, BERT_colors)):
        widths = np.array([len(token)])
        starts = BERT_token_lens_cum[i] - widths
        ax.barh(BERT_bar_height, widths, left=starts, color=color, height=0.8)
        xcenters = starts + widths / 2
    
        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'black'
        ax.text(xcenters, BERT_bar_height-0.1, token, ha='center', va='center',
                color=text_color, size=18)
        ax.text(xcenters, BERT_bar_height+0.2, str(round(BERT_similarity[i], 3)), ha='center', va='center',
                color=text_color, size=12)
    
    # plot merged tokens
    merged_bar_height = figheight/4*3-0.2
    
    for i, (token, color) in enumerate(zip(merged_tokens, merged_colors)):
        widths = np.array([len(token) * ratio])
        starts = merged_token_lens_cum[i] - widths
        ax.barh(merged_bar_height, widths, left=starts, color=color, height=0.8)
        xcenters = starts + widths / 2
    
        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'black'
        ax.text(xcenters, merged_bar_height-0.1, token, ha='center', va='center',
                color=text_color, size=18)
        ax.text(xcenters, merged_bar_height+0.2, str(round(merged_similarity[i], 3)), ha='center', va='center',
                color=text_color, size=12)
        
    ax.text(0, figheight*0.05, "Sample: "+sample_obj['original_sentence'], ha='left', va='top',
            color='black', size=15)
    ax.text(0, figheight*0.95, "Keywords: "+sample_obj['keywords'], ha='left', va='bottom',
            color='black', size=15)
    
    plt.show()
    
def merged_BERT_tokens(BERT_tokens, BERT_token_embeddings, sentence_embedding):
    
    merged_tokens, merged_simi, merged_embedding_list = [], [], []
    prev_token = BERT_tokens[0]
    prev_token_embedding = [BERT_token_embeddings[0]]
    
    for curr_idx in range(1, len(BERT_tokens)):
        
        curr_token = BERT_tokens[curr_idx]
        curr_token_embedding = BERT_token_embeddings[curr_idx]
        
        if curr_token[:2] == '##':
            prev_token += curr_token[2:]
            prev_token_embedding.append(curr_token_embedding)
            continue
        else:
            merged_tokens.append(prev_token)
            merged_embeddings = torch.mean(torch.stack(prev_token_embedding), dim=0)
            merged_embedding_list.append(merged_embeddings)
            merged_simi.append(1-cosine(merged_embeddings, sentence_embedding))
            
            prev_token = curr_token
            prev_token_embedding = [curr_token_embedding]
            
    merged_tokens.append(prev_token)
    merged_embeddings = torch.mean(torch.stack(prev_token_embedding), dim=0)
    merged_embedding_list.append(merged_embeddings)
    merged_simi.append(1-cosine(merged_embeddings, sentence_embedding))
    
    return merged_tokens, merged_simi, merged_embedding_list

def cal_merged_simi(tensors, sentence_embedding):
    if len(tensors) > 1:
        merged_tensor = torch.mean(torch.stack(tensors), dim=0)
    else:
        merged_tensor = tensors[0]
    return 1-cosine(merged_tensor, sentence_embedding)

def get_keyphrase(simi_lst):
    
    """
    extract the 
    """