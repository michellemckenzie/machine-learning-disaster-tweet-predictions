from concurrent.futures import process
import numpy as np
import pandas as pd
import re

# NLP Packages
from nltk.corpus import stopwords
from collections import Counter
import string

import matplotlib.pyplot as plt

import nltk
from nltk.stem import WordNetLemmatizer
from nltk import WhitespaceTokenizer
from nltk.corpus import wordnet
nltk.download('averaged_perceptron_tagger')
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pickle

svm = 'finalized_model.sav'
tfidf = 'tfidf_model.sav'

svm_model = pickle.load(open(svm, 'rb'))
tfidf_model = pickle.load(open(tfidf, 'rb'))

# Lemmatize function

word_tokenizer = nltk.tokenize.WhitespaceTokenizer()
wordnetlemmatizer = nltk.WordNetLemmatizer()
pos_letters = ['V', 'R', 'J', 'N'] # possible first letters of pos

def pos_functions(pos):
    if pos.startswith(pos_letters[0]): # seeing if POS is a verb
        return wordnet.VERB # corresponding verb function
    elif pos.startswith(pos_letters[1]): # seeing if POS is an adverb
        return wordnet.ADV # corresponding adverb function
    elif pos.startswith(pos_letters[2]): # seeing if POS is an adjective
        return wordnet.ADJ # corresponding adjective function
    elif pos.startswith(pos_letters[3]): # seeing if POS is a noun
        return wordnet.NOUN # corresponding noun function
    else:
        return None


def lemmatize_text(input):
    new_text = [] # initializing new list to store lemmatized words

    sentence_words = nltk.word_tokenize(input) # splitting up sentence into words
    pos_sentence_words = nltk.pos_tag(sentence_words) # getting the pos for each word

    for word, pos in pos_sentence_words:
        if pos_functions(pos) is None:  # may not be pos tag
            new_text.append(word) # adding original word to new list if no pos tag
        else: 
            new_lemmatized = wordnetlemmatizer.lemmatize(word, pos_functions(pos)) # passing corresponding pos function to lemmatizer
            new_text.append(new_lemmatized) # adding lemmatized word to list is there'ss a pos tag
    return " ".join(new_text) # getting rid of the commas between each word in the list


# This is where the data cleaning and lemmatiztion occurs

# I will replace all these punctuation with ''
punctuation = '["\'?,\.!=><#$%^&*()-+@[]\{\}_~`]'
# Add any words you want to be replaced and the conversion
abbr_dict = {
    "'til": "until",
    "'ll": " will",
    "y'all": "you all",
    "that's": "that is",
    "won't": "will not",
    "can't": "can not",
    "cannot": "can not",
    "ain't": "am not",
    "n't": " not",  # wasn't weren't haven't didn't
    "'ve": " have",  # i've we've you've would've
    "'d": " would",  # you'd i'd we'd you'd
    "'re": " are",  # they're we're you're
    "in'": "ing",  # lookin'

    "what's": "what is",
    "what're": "what are",
    "who's": "who is",
    "who're": "who are",
    "where's": "where is",
    "where're": "where are",
    "when's": "when is",
    "when're": "when are",
    "how's": "how is",
    "how're": "how are",

    "i'm": "i am",
    "we're": "we are",
    "you're": "you are",
    "they're": "they are",
    "it's": "it is",
    "he's": "he is",
    "she's": "she is",
    "that's": "that is",
    "there's": "there is",
    "there're": "there are",

    "i've": "i have",
    "we've": "we have",
    "you've": "you have",
    "they've": "they have",
    "who've": "who have",
    "would've": "would have",
    "not've": "not have",

    "i'll": "i will",
    "we'll": "we will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "it'll": "it will",
    "they'll": "they will",

    "isn't": "is not",
    "wasn't": "was not",
    "aren't": "are not",
    "weren't": "were not",
    "can't": "can not",
    "couldn't": "could not",
    "don't": "do not",
    "dont": "do not",
    "didn't": "did not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "doesn't": "does not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "won't": "will not",
    "#": "",
    punctuation: '',
    '\s+': ' ',  # replace multi space with one single space

}
adjectives = {
    " angry ": ' ',
    " brave ": ' ',
    " careful ": ' ',
    " healthy ": ' ',
    " little ": ' ',
    " old ": ' ',
    " generous ": ' ',
    " tall ": ' ',
    " good ": ' ',
    " big ": ' ',
    " small ": ' ',
    " am ": ' ',
}
pronouns = {
    " i ": '',
    " he ": '',
    " she ": '',
    " its ": '',
    " me ": '',
    " my ": '',
    " that ": '',
    " they ": '',
    " this ": '',
    " those ": '',
    " im ": '',
    " u ": ''
}
articles = {
    " the ": ' ',
    " a ": ' ',
    " an ": ' ',
}

# Reads the data, converts to lowercase, and replaces using abbreviation list


def process_data(input):
    input = input.lower()  # convert tweet to lower case
    input = re.sub('\d+', '', input)  # remove numbers
    # data.keyword = data.keyword.str.lower()  # convert keyword to lower case
    #data.keyword = data.keyword.astype(str)
    for word, replacement in abbr_dict.items():
        input = re.sub(word, replacement, input)

    for word, replacement in pronouns.items():
        input = re.sub(word, replacement, input)
    for word, replacement in articles.items():
        input = re.sub(word, replacement, input)

    # remove http or https links from tweets
    input = re.sub(r"http\S+", "", input)
    input = re.sub('[^\w\s]', '', input)
    return input


# Example


def takeInput(text):
    text = process_data(text)
    text = lemmatize_text(text)

    transformedText = tfidf_model.transform([text])

    # return test
    print(svm_model.predict(transformedText)[0])
    return svm_model.predict(transformedText)[0]
