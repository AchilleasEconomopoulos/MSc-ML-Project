'''A module for all the utility functions and tools used in the Notebooks.'''

import pandas as pd
from spellchecker import SpellChecker
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack
import os


ps = PorterStemmer()        # Stemming
spell = SpellChecker()      # Spellchecking (for inference)
wnl = WordNetLemmatizer()   # Lemmatization (not used)

def load_csv(filepath,lexicon=True,pos=False):
    '''Simple method that loads the (intended) csv
       and returns all raw texts and their labels (ground truth). '''

    # dirname = os.path.dirname(__file__)
    # filepath = os.path.join(dirname, filename)

    df = pd.read_csv(filepath)
    texts = df['text'].values
    labels = df['label'].values
    
    # TODO: Return pos features and validate with them
    # pos_features = df['pos_features'].values

    if(lexicon):
        scores = df['avg_lexicon_score'].values
        return texts,labels,scores
    
    else:
        return texts, labels

    
def custom_tokenizer(text):
    '''Tokenization using a regex.'''
    # vectorizer_default = r"(?u)\b\w\w+\b"
    first_person = r"I\'m|i\'m|I|i" 
    vectorizer_default = r"(?u)\b\w\w+\b\'?\w*" 
    number_elimination = r"\b\w*\d\w*\b"
    underscore_elimination = r"_"

    tokens = re.findall(f"{vectorizer_default}|{first_person}", text) # Get the tokens matching the base regex
    tokens = [t for t in tokens if not re.match(f'{number_elimination}|{underscore_elimination}', t)] # Eliminate number and underscore tokens
    tokens = [ps.stem(token) for token in tokens] # Stem what's left
    # tokens = [wnl.lemmatize(token) for token in tokens]
    return tokens

def spellcheck(tokens):
    '''Spellchecking of the tokens.'''
    spellchecked_tokens = []
    for token in tokens:
        spellchecked_token = spell.correction(token)
        if(spellchecked_token):
            spellchecked_tokens.append(spellchecked_token)
        else:
            spellchecked_tokens.append(token)
    return pd.array(spellchecked_tokens)


# def extract_pos_features(text):
#     """Extracts % of nouns, verbs, adjectives, adverbs."""
#     doc = nlp(text)
#     total_tokens = len(doc)
    
#     pos_counts = {
#         "NOUN": sum(1 for token in doc if token.pos_ == "NOUN"),
#         "VERB": sum(1 for token in doc if token.pos_ == "VERB"),
#         "ADJ": sum(1 for token in doc if token.pos_ == "ADJ"),
#         "ADV": sum(1 for token in doc if token.pos_ == "ADV")
#     }
    
#     # Normalize counts to percentages
#     return np.array([pos_counts["NOUN"], pos_counts["VERB"], pos_counts["ADJ"], pos_counts["ADV"]]) / total_tokens

def transform_input(input, vectorizer: TfidfVectorizer, scaler: MinMaxScaler):
    '''
    Transforms input for inference:
    1) spellchecking
    2) tokenization (using trained tokenizer)
    3) lexicon score calculation and normalization (using trained scaler)
    '''

    if(type(input)==str):
        input = [input] # Convert to list

    for idx, text in enumerate(input):
        tokens = text.split()
        spellchecked_tokens = spellcheck(tokens)
        input[idx] = ' '.join(spellchecked_tokens)

    X_in = vectorizer.transform(input)

    lex_score = score_input_with_lexicon(input)

    # Normalize the lexicon scores to bring them in line with the tfidf features
    normalized_lex_score = scaler.transform(np.array(lex_score).reshape(-1,1))
    
    X_in = hstack([X_in, normalized_lex_score])

    X_in = pd.DataFrame(X_in.toarray(), columns=np.append(vectorizer.get_feature_names_out(),"lexicon_score"))
    
    return X_in

def score_input_with_lexicon(texts):
    lexicon_df = _get_lexicon()

    lexicon_scores = []                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
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

    return lexicon_scores

def _get_lexicon():
    dirname = os.path.dirname(__file__)
    lexicon_path = os.path.join(dirname, "../data/worrywords-v1.txt")

    with open(lexicon_path, 'r') as f:
        worrywords = f.read().splitlines()
        cols = worrywords[0].split('\t')

    words = {'Term':[], 'Mean':[]}
    for line in worrywords[1:]:
        cols = line.split('\t')
        words['Term'].append(cols[0])
        words['Mean'].append(float(cols[1]))

    # Convert to dataframe for easier access
    lexicon_df = pd.DataFrame(words)

    return lexicon_df


############## ATTEMPT AT "CUSTOM" IMPLEMENTATIONS ##############

# def stemming(tokens):
#     '''Stemming of the tokens.'''
#     stemmed_tokens = []
#     for token in tokens:
#         spellchecked_token = token
#         if(spellchecked_token):
#             stemmed_tokens.append(ps.stem(spellchecked_token))
#         else:
#             stemmed_tokens.append(ps.stem(token))

#     return pd.array(stemmed_tokens)


# class TFIDFCalculator:
#     '''A set of methods to convert raw text samples of a dataset
#        into TF-IDF values for all the words it contains.'''

#     corpus_tokens = {}

#     def __init__(self, filepath):
#         '''Constructs an instance based on a file / dataset'''
#         self.texts, self.labels = load_csv(filepath)

#     def extract_tf_from_document(self,document_tokens):
#         '''Extracts the term frequencies from a token set.'''

#         document_freqs = {}
#         document_tfs = {}
#         total_words_in_document = 0

#         for token in document_tokens:
#             total_words_in_document += 1

#             if token not in document_freqs:
#                 document_freqs[token] = 0
#             document_freqs[token] += 1

#         for token in document_freqs.keys():
#             document_tfs[token] = document_freqs[token] / total_words_in_document

#         return document_tfs


#     def update_token_tfs(self, tf_dict):
#         '''Updates the term frequencies of the corpus tokens.'''

#         for token in tf_dict.keys():
#             if token not in self.corpus_tokens:
#                 self.corpus_tokens[token] = {}
#                 self.corpus_tokens[token]['tf'] = 0.0
#                 self.corpus_tokens[token]['document_count'] = 0
            
#             self.corpus_tokens[token]['tf'] += tf_dict[token]
#             self.corpus_tokens[token]['document_count'] += 1


#     def update_token_idfs(self):
#         '''Updates the inverse document frequencies of the corpus tokens.'''
#         num_of_documents = len(self.texts)

#         for token, counters in self.corpus_tokens.items():
#             self.corpus_tokens[token]['idf'] = np.log(num_of_documents / (counters['document_count']))
#             self.corpus_tokens[token]['tfidf'] = np.multiply(counters['tf'], counters['idf'])


#     def get_features(self):
#         '''Calculates the TF-IDF values for all the words in the corpus.'''
#         i = 0
#         for document in self.texts:
#             i += 1
#             print("document " + str(i))
#             document_tokens = tokenize(document)
#             stemmed_tokens = spellcheck_and_stem(document_tokens)
            
#             term_frequencies = self.extract_tf_from_document(stemmed_tokens)
#             self.update_token_tfs(term_frequencies)

#         self.update_token_idfs()

#         with open ('tfidf2.json', 'w') as f:
#             json.dump(self.corpus_tokens, f, indent=4)