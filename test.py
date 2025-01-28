from utils.data_loader import TFIDFCalculator, spellcheck_and_stem

# texts, labels = load_csv('data/dreaddit-train.csv')

# tokens = tokenize(texts[0])
# stemmed_tokens = stem_spellcheck(tokens)

# print(tokens)
# print(stemmed_tokens)

calculator = TFIDFCalculator('data/dreaddit-train.csv')

calculator.get_features()

# print(calculator.get_all_tfs())

# print(spellcheck_and_stem(["Hello", "native", "congratulations", "congratulate"]))