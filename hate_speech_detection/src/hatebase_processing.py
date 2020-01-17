import numpy as np
import pandas as pd
from os import getcwd, listdir
import nltk
from nltk.stem.porter import *
import re

data_path = getcwd() + "/hate_speech_detection/data/HatebaseTwitter"
listdir(data_path)
df = pd.read_csv(f"{data_path}/{listdir(data_path)[1]}")
df.head()

# To Install the Stopwords with NLTK:
nltk.download('stopwords')

stopwords = nltk.corpus.stopwords.words('english')

other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)

stemmer = PorterStemmer()

def preprocess(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE
    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
    #parsed_text = parsed_text.code("utf-8", errors='ignore')
    return parsed_text



tweets = df['tweet'].tolist()

def tokenize(tweet):
    """Removes punctuation & excess whitespace, sets to lowercase,
    and stems tweets. Returns a list of stemmed tokens."""
    tweet = " ".join(re.split("[^a-zA-Z]*", tweet.lower())).strip()
    #tokens = re.split("[^a-zA-Z]*", tweet.lower())
    tokens = [stemmer.stem(t) for t in tweet.split()]
    return tokens

tokenize(tweets[0])
