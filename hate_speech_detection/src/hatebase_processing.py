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
