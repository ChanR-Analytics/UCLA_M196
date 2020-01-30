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

# Performing Machine Learning Using Dimensionality Reduction Techniques and Supervised Classification Techniques 
X_ = hb.l1_dim_reduce(features)
X_ = hb.rfe_dim_reduce(X_, 'ada', 20)
multi_classifier = hb.classify(X_, 'binary', 'xgb', 0.15, res=True, res_method='up')
