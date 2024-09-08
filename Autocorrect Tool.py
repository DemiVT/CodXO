from nltk.corpus import words
from nltk.metrics.distance import edit_distance
import nltk

nltk.download('words')
word_list = set(words.words())

def autocorrect(word):
    candidates = [w for w in word_list if edit_distance(word, w) < 3]
    return min(candidates, key=lambda x: edit_distance(word, x)) if candidates else word

# Example usage
word = 'speling'
corrected_word = autocorrect(word)
print(f'Corrected word for "{word}": {corrected_word}')
