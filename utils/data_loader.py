import json
import pandas as pd
from spellchecker import SpellChecker
from nltk.stem import PorterStemmer
import numpy as np
import os
import re

ps = PorterStemmer()
spell = SpellChecker()

def load_csv(filepath):
    '''Simple method that loads the (intended) csv
       and returns all raw texts and their labels (ground truth). '''

    # dirname = os.path.dirname(__file__)
    # filepath = os.path.join(dirname, filename)

    df = pd.read_csv(filepath)
    texts = df['text'].values
    labels = df['label'].values
    return texts, labels

    
def tokenize(text):
    '''Tokenization using a regex.'''
    tokens = pd.array(re.split(r'[;,.!?\s"[0-9]()]+', text))
    return tokens

def spellcheck_and_stem(tokens):
    '''Spellcheck and stemming of the tokens.'''
    stemmed_tokens = []
    for token in tokens:
        spellchecked_token = token
        if(spellchecked_token):
            stemmed_tokens.append(ps.stem(spellchecked_token))
        else:
            stemmed_tokens.append(ps.stem(token))

    return pd.array(stemmed_tokens)

class TFIDFCalculator:
    '''A set of methods to convert raw text samples of a dataset
       into TF-IDF values for all the words it contains.'''

    corpus_tokens = {}

    def __init__(self, filepath):
        '''Constructs an instance based on a file / dataset'''
        self.texts, self.labels = load_csv(filepath)

    def extract_tf_from_document(self,document_tokens):
        '''Extracts the term frequencies from a token set.'''

        document_freqs = {}
        document_tfs = {}
        total_words_in_document = 0

        for token in document_tokens:
            total_words_in_document += 1

            if token not in document_freqs:
                document_freqs[token] = 0
            document_freqs[token] += 1

        for token in document_freqs.keys():
            document_tfs[token] = document_freqs[token] / total_words_in_document

        return document_tfs


    def update_token_tfs(self, tf_dict):
        '''Updates the term frequencies of the corpus tokens.'''

        for token in tf_dict.keys():
            if token not in self.corpus_tokens:
                self.corpus_tokens[token] = {}
                self.corpus_tokens[token]['tf'] = 0.0
                self.corpus_tokens[token]['document_count'] = 0
            
            self.corpus_tokens[token]['tf'] += tf_dict[token]
            self.corpus_tokens[token]['document_count'] += 1


    def update_token_idfs(self):
        '''Updates the inverse document frequencies of the corpus tokens.'''
        num_of_documents = len(self.texts)

        for token, counters in self.corpus_tokens.items():
            self.corpus_tokens[token]['idf'] = np.log(num_of_documents / (counters['document_count']))
            self.corpus_tokens[token]['tfidf'] = np.multiply(counters['tf'], counters['idf'])


    def get_features(self):
        '''Calculates the TF-IDF values for all the words in the corpus.'''
        i = 0
        for document in self.texts:
            i += 1
            print("document " + str(i))
            document_tokens = tokenize(document)
            stemmed_tokens = spellcheck_and_stem(document_tokens)
            
            term_frequencies = self.extract_tf_from_document(stemmed_tokens)
            self.update_token_tfs(term_frequencies)

        self.update_token_idfs()

        with open ('tfidf2.json', 'w') as f:
            json.dump(self.corpus_tokens, f, indent=4)


                


        

        
                 
                 
    



