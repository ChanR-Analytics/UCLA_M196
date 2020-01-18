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

class HatebaseTwitter():
    def __init__(self, data_path):
        # Reading the Data Frame from the Hatebase Twitter CSV File
        self.df = pd.read_csv(f"{data_path}/{listdir(data_path)[0]}")

    def eda(self):
        # Seeing the Number of CrowdFlower Annotators that Decided On a Tweet's Label
        
