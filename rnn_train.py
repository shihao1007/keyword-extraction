#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 16:04:00 2020

@author: shihao
"""


import tensorflow as tf
import numpy as np
import tw_funcs as tw
import datetime

from gensim.models import KeyedVectors
#%%

X_train = np.load('dataset/X_train.npy', allow_pickle=True)
y_train = np.load('dataset/y_train_binary.npy', allow_pickle=True)
X_test = np.load('dataset/X_test.npy', allow_pickle=True)
y_test = np.load('dataset/y_test_binary.npy', allow_pickle=True)

X_train = tf.keras.preprocessing.sequence.pad_sequences(
    X_train, maxlen=25, dtype='float32', padding='post', truncating='post',
    value=0.0
)

X_test = tf.keras.preprocessing.sequence.pad_sequences(
    X_test, maxlen=25, dtype='float32', padding='post', truncating='post',
    value=0.0
)

y_train = tf.keras.preprocessing.sequence.pad_sequences(
    y_train, maxlen=25, dtype='float32', padding='post', truncating='post',
    value=0.0
)

y_test = tf.keras.preprocessing.sequence.pad_sequences(
    y_test, maxlen=25, dtype='float32', padding='post', truncating='post',
    value=0.0
)
#%%
model = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(25, activation='softmax')
])

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(1e-4),
    metrics=['accuracy']
)
#%%
history = model.fit(
    X_train, y_train, 
    batch_size=32, 
    epochs=4, 
    validation_split=0.2)

model.save('models/BiLSTM_'+datetime.datetime.now().strftime('%Y-%m-%d--%H-%M-%S'))

tw.plot_graphs(history, 'accuracy')

test_loss, test_acc = model.evaluate(X_test, y_test)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))
