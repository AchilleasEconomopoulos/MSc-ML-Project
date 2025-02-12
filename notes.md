**Problem:** Spellchecking creates bottleneck for feature extraction

**Solution:**
1) Extract ALL possible tokens from the corpus (using some standard delimiters) and produce an intermediate vocabulary that won't need spellchecking again.
2) Process the intermediate vocabulary to produce a more refined one in terms of it having valid words (delete numbers, single letters, etc.).
3) Implement the TFIDF calculation on the refined vocabulary. This way the calculation itself can be tweaked and refined without major delays.