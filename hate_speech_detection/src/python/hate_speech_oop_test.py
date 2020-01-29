import pandas as pd
from os import getcwd, listdir
from hate_speech_detection.src.python.hate_speech_oop import HatebaseTwitter


# Getting Davidson's Hatbase Twitter Data
hb_path = getcwd() + "/hate_speech_detection/data/HatebaseTwitter"

# Initializing the HatebaseTwitter Class
hb = HatebaseTwitter(hb_path)

# Performing an Exploratory Data Analysis of the Hatebase Twitter Dataset
hb.eda()

# Extracting the Tweet TF-IDF, POS TF-IDF, and Other Tweet Features into a Multidimensional Data Matrix
features = hb.features()

X_ = hb.multi_l1_dim_reduce(features)
X_ = hb.multi_rfe_dim_reduce(X_, 10)
X_.shape 
classifier = hb.multi_classify(X_, hb.df['class'], 'xgb', 0.15)
