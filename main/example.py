import random

def lookup(words,w):
    w = w.lower()
    if w in words:
        return words[w]
    else:
        return words['UUUNKKK']

class example(object):

    def __init__(self, phrase):
        self.phrase = phrase.strip()
        self.embeddings = []
        self.representation = None

    def populate_embeddings(self, words):
        phrase = self.phrase.lower()
        arr = phrase.split()
        for i in arr:
            self.embeddings.append(lookup(words,i))

    def populate_embeddings_scramble(self, words):
        phrase = self.phrase.lower()
        arr = phrase.split()
        random.shuffle(arr)
        for i in arr:
            self.embeddings.append(lookup(words,i))

    def unpopulate_embeddings(self):
        self.embeddings = []
