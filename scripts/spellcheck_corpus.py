'''
A script that spellchecks the entirety of the train and test sets,
and saves them to a new file.
'''

import os
import sys
import csv

# Add utils folder to syspath in order to import the util functions
dirname = os.path.dirname(__file__)
parent_dir = sys.path.append(os.path.abspath(os.path.join(dirname, '..')))

from utils.data_loader import load_csv, spellcheck

# Get the data dir path relative to the current file
data_dir = os.path.join(dirname, '../data')

# Spellcheck either the train or the test dataset
# by commenting out the corresponding pair of lines

# TRAIN
texts,labels = load_csv(data_dir + '/dreaddit-train.csv')
new_filepath = data_dir + '/dreaddit-train-spellchecked.csv'

# TEST
# texts, labels = load_csv(data_dir + '/dreaddit-test.csv')
# new_filepath = data_dir + '/dreaddit-test-spellchecked.csv'

# Spellcheck the tokens of each sample and then recreate it.
with open(new_filepath, 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['text', 'label'])
    for i in range(len(texts)):
        tokens = texts[i].split()
        spellchecked_tokens = spellcheck(tokens)
        spellchecked_text = ' '.join(spellchecked_tokens)
        writer.writerow([spellchecked_text, labels[i]])

