import pandas as pd
from spellchecker import SpellChecker
from nltk.stem import PorterStemmer
import numpy as np
import re


from utils.data_loader import tokenize, spellcheck_and_stem

vocab = []
condition = r'\([0-9]+f\)|\([0-9]+m\)|(^[a-zA-Z0-9]$)'
# with open('initial_vocab.txt', 'r') as f:
#     for line in f:
#         if not re.match(r'\([0-9]+f\)|\([0-9]+m\)', line):
#             vocab.append(line.strip())

# print(vocab)

print(re.match(condition,' (28f)'))

# tokens = ['word1','word2','(28f)']

# for token in tokens:
#     if token 
    