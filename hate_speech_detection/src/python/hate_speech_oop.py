import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
import nltk
from nltk.stem.porter import *
from os import getcwd, listdir
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
from textstat.textstat import *
warnings.simplefilter(action='ignore', category=FutureWarning)


class HatebaseTwitter():
    def __init__(self, data_path):
        # Reading the Data Frame from the Hatebase Twitter CSV File
        self.df = pd.read_csv(f"{data_path}/{listdir(data_path)[0]}")

    def eda(self):
        # Seeing the Number of CrowdFlower Annotators that Decided On a Tweet's Label
        annot_count = self.df['count'].value_counts().to_dict()

        ## Visualizing the Annotator Count
        plt.figure(figsize=(12,12))
        plt.bar(list(annot_count.keys()), list(annot_count.values()))
        plt.xlabel('Number of CrowdFlower Annotators', size=14)
        plt.ylabel('Number of Tweets Annotated', size=14)

        # Visualizing the Proportion of Tweet Classification Labels and Class Imbalance
        plt.figure(figsize=(12,12))
        self.df['class'].plot_pie() 
