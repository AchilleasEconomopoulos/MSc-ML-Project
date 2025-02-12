from utils.data_loader import TFIDFCalculator, spellcheck_and_stem
import json
# texts, labels = load_csv('data/dreaddit-train.csv')

# tokens = tokenize(texts[0])
# stemmed_tokens = stem_spellcheck(tokens)

# print(tokens)
# print(stemmed_tokens)

# calculator = TFIDFCalculator('data/dreaddit-train.csv')

# calculator.get_features()

with open('tfidf.json', 'r') as f:
    vocab = json.load(f)

with open('vocab.txt', 'w') as f:
    for word in vocab:
        f.write(word + '\n')

# print(calculator.get_all_tfs())

# print(spellcheck_and_stem(["Hello", "native", "congratulations", "congratulate"]))