import nltk
import pandas as pd
import numpy as np

from nltk.classify import NaiveBayesClassifier as nb
from nltk.classify.util import accuracy

file = open('SMSSpamCollection.txt')

data = []
for line in file:
    data.append(line)

def format_sentence(sent):
  return({word: True for word in nltk.word_tokenize(sent)})

ham = []
spam = []

for string in data:
    if string.startswith('ham'):
        sentence = string.split('\t')[1]
        ham.append([format_sentence(sentence), 'ham'])
    elif string.startswith('spam'):
        sentence = string.split('\t')[1]
        spam.append([format_sentence(sentence), 'spam'])

training = ham[:int((0.8)*len(ham))] + spam[:int((0.8)*len(spam))]
test = ham[int((0.8)*len(ham)):] + spam[int((0.8)*len(spam)):]

classifier = nb.train(training)
classifier.show_most_informative_features()

print(accuracy(classifier, test)) ## Accuracy score of 0.92
