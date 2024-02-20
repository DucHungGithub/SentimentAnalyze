'''
Note: This folder is set as the default on the D: drive to run
'''


import re
from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import math
import sys, os, csv, heapq

import os
import pandas as pd


print(punctuation)
import py_vncorenlp

# Automatically download VnCoreNLP components from the original repository
# and save them in some local machine folder
# py_vncorenlp.download_model(save_dir='D:\\preprocessing\\check\\vncorenlp')

# # # Load the word and sentence segmentation component
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='D:\\preprocessing\\check\\vncorenlp')


def preprocess(document):
    lower_process = document.lower()
    punctuation_process = ""
    for i in lower_process:
        if i not in punctuation:
            punctuation_process += i

    white_space_process = ' '.join(punctuation_process.split())

    output = rdrsegmenter.word_segment(white_space_process)

    return output

# For train
test_data = pd.read_csv("D:\\preprocessing\\train_data.csv")
print(test_data)
test_sentences = test_data['sentence']
word_process = []

word_label = []

for line in test_sentences:
    temp = preprocess(line)[0] + '\n'
    word_process.append(temp)

for line in test_data['sentiment']:
    temp = str(line) + '\n'
    word_label.append(temp)
    
with open('D:\\preprocessing\\trainsentsolve.txt', 'w+',encoding='UTF-8') as f:
    f.writelines(word_process)

with open('D:\\preprocessing\\trainsentlabel.txt', 'w+',encoding='UTF-8') as f:
    f.writelines(word_label)
    
# For test
test_data = pd.read_csv("D:\\preprocessing\\test_data.csv")
print(test_data)
test_sentences = test_data['sentence']
word_process = []

word_label = []

for line in test_sentences:
    temp = preprocess(line)[0] + '\n'
    word_process.append(temp)

for line in test_data['sentiment']:
    temp = str(line) + '\n'
    word_label.append(temp)
    
with open('D:\\preprocessing\\testsentsolve.txt', 'w+',encoding='UTF-8') as f:
    f.writelines(word_process)

with open('D:\\preprocessing\\testsentlabel.txt', 'w+',encoding='UTF-8') as f:
    f.writelines(word_label)
    
# For valid
test_data = pd.read_csv("D:\\preprocessing\\validation_data.csv")
print(test_data)
test_sentences = test_data['sentence']
word_process = []

word_label = []

for line in test_sentences:
    temp = preprocess(line)[0] + '\n'
    word_process.append(temp)

for line in test_data['sentiment']:
    temp = str(line) + '\n'
    word_label.append(temp)
    
with open('D:\\preprocessing\\validsentsolve.txt', 'w+',encoding='UTF-8') as f:
    f.writelines(word_process)

with open('D:\\preprocessing\\validsentlabel.txt', 'w+',encoding='UTF-8') as f:
    f.writelines(word_label)