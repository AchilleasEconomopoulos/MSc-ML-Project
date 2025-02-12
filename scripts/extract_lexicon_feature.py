'''
A script to calculate a new feature for each sample in the dataset, based on an anxiety lexicon,
and create an updated csv file with the new feature.
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

# Choose either train or test set
# dataset = 'dreaddit-train-spellchecked'
dataset = 'dreaddit-test-spellchecked'

old_filepath = os.path.join(data_dir, f'{dataset}.csv')
new_filepath = os.path.join(data_dir, f'{dataset}-lex.csv')

texts,labels = load_csv(old_filepath,False)


# Get the words and their "scores" from the lexicon
with open(os.path.join(data_dir,'worrywords-v1.txt'), 'r') as f:
    worrywords = f.read().splitlines()
    cols = worrywords[0].split('\t')

words = {'Term':[], 'Mean':[]}
for line in worrywords[1:]:
    cols = line.split('\t')
    words['Term'].append(cols[0])
    words['Mean'].append(float(cols[1]))

# Convert to dataframe for easier access
lexicon_df = pd.DataFrame(words)

lexicon_scores = []
# Calculate the average "score" for each word in the text
for text in texts:
    lexicon_score = 0
    count = 0

    # Split the text into words and calculate the document's total score according to the lexicon
    for word in text.split():
        word = str(word).lower()
        if word in lexicon_df['Term'].values:
            lexicon_score += lexicon_df.loc[lexicon_df['Term'] == word, 'Mean'].values[0]
            count += 1

    # Calculate the average (because of high variance on sample lengths)
    avg_lexicon_score = 0
    if count > 0:
        avg_lexicon_score = lexicon_score / count
    lexicon_scores.append(avg_lexicon_score)




# "Update" the old dataset so that the feature extraction process is not repeated for training.
df = pd.read_csv(old_filepath)
df['avg_lexicon_score'] = lexicon_scores
df.to_csv(new_filepath, index=False)