import pandas as pd
import numpy as np

# NLP Packages
from nltk.corpus import stopwords
from textblob import TextBlob
from textblob import Word
import string
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

import re

punctuation = '["\'?,\.!=><#$%^&*()-+@[]\{\}_~`]'  # I will replace all these punctuation with ''
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
    " i ": ' ',
    " he ": ' ',
    " she ": ' ',
    " its ": ' ',
    " me ": ' ',
    " my ": ' ',
    " that ": ' ',
    " they ": ' ',
    " this ": ' ',
    " those ": ' ',
}

articles = {
    " the ": ' ',
    " a ": ' ',
    " an ": ' ',
}

# Reads the data, converts to lowercase, and replaces using abbreviation list


def process_data(file_name):
    data = pd.read_csv(file_name)
    data.drop(['id','location'], inplace = True)
    data.text = data.text.str.lower()  # convert tweet to lower case
    data.text = data.text.str.replace('\d+', '') #remove numbers
    data.keyword = data.keyword.str.lower() #convert keyword to lower case
    data.text = data.text.astype(str) #cast tweet as a string
    data.keyword = data.keyword.astype(str)
    data.replace(abbr_dict, regex=True, inplace=True)
    data.replace(pronouns, regex=True, inplace=True)
    data.replace(articles, regex=True, inplace=True)

     # remove http or https links from tweets
    for sentence in range(len(data['text'])):
        data['text'][sentence] = re.sub(r"http\S+", "", data['text'][sentence])
 
    return data
