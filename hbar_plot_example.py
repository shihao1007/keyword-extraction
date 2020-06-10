#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 15:22:52 2020

@author: shihao
"""


import numpy as np
import matplotlib.pyplot as plt

tokens = ['[CLS]', 'swat', 'and', 'sheriff', "'", 's', 'deputies', 'conducting', 'investigation', 'on', 'mum', '##ph', '##ord', 'street', 'in', 'victoria', ',', 'tx', ':', 'local', 'source', 'photo', '.', '[SEP]']

original_sentence = " ".join(tokens)

importance = np.array([0.5082683563232422,
 0.5039727091789246,
 0.6312222480773926,
 0.5234028697013855,
 0.29067179560661316,
 0.6373759508132935,
 0.6470983624458313,
 0.6684234142303467,
 0.6152777075767517,
 0.6382220387458801,
 0.4094296991825104,
 0.4067043364048004,
 0.42847567796707153,
 0.5405606031417847,
 0.5765551328659058,
 0.3278561234474182,
 0.33367231488227844,
 0.35320526361465454,
 0.5775998830795288,
 0.6211853623390198,
 0.600104033946991,
 0.5414994359016418,
 0.5419120192527771,
 0.21641527116298676])

#%%
importance = np.array(importance)
token_lens = np.array([len(t) for t in tokens])
token_lens_cum = token_lens.cumsum()
norm = importance/np.max(importance)
colors = plt.get_cmap('GnBu')(norm)

figheight = 2
figwidth = 30

fig, ax = plt.subplots(figsize=(figwidth, figheight))
ax.invert_yaxis()
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.set_ylim(figheight)

for i, (token, color) in enumerate(zip(tokens, colors)):
    widths = np.array([len(token)])
    starts = token_lens_cum[i] - widths
    ax.barh(figheight/2, widths, left=starts, color=color, height=0.8)
    xcenters = starts + widths / 2

    r, g, b, _ = color
    text_color = 'white' if r * g * b < 0.5 else 'black'
    ax.text(xcenters, figheight/2, token, ha='center', va='center',
            color=text_color, size=20)
    
ax.text(0, 0.2, original_sentence, ha='left', va='top',
        color='black', size=15)
ax.text(0, 1.8, original_sentence, ha='left', va='bottom',
        color='black', size=15)

plt.show()