'''
TODO
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import sys

# Add utils folder to syspath in order to import the util functions
dirname = os.path.dirname(__file__)
parent_dir = sys.path.append(os.path.abspath(os.path.join(dirname, '..')))

from utils.data_loader import extract_pos_features, load_csv

# Get the data dir path relative to the current file
data_dir = os.path.join(dirname, '../data')

# Lexicon scores for the train set only
dataset = 'dreaddit-train-spellchecked'
dataset = 'dreaddit-test-spellchecked'

old_filepath = os.path.join(data_dir, f'{dataset}.csv')
new_filepath = os.path.join(data_dir, f'{dataset}-2.csv')

texts,labels = load_csv(old_filepath,False)


pos_features = []
# Calculate the average "score" for each word in the text
for text in texts:

    pos_features.append(extract_pos_features(text))


# "Update" the old dataset so that the feature extraction process is not repeated for training.
df = pd.read_csv(old_filepath)
df['pos_features'] = pos_features
df.to_csv(new_filepath, index=False)