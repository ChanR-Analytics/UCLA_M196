import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from os import getcwd, listdir
from jupyterthemes import jtplot
from sklearn.feature_extraction.text import TfidfVectorizer
sns.set()
jtplot.style(theme="monokai")

class HatebaseTwitter_Exploration():
    def __init__(self, data_path):
