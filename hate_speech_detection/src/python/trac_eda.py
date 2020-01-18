from os import getcwd, listdir
import numpy as np
import pandas as pd

# HateBase Twitter Data
data_path = getcwd() + "/hate_speech_detection/data/HatebaseTwitter/english"
listdir(data_path)
english_data_dict = {i[:-4]: pd.read_csv(f"{data_path}/{i}") for i in listdir(data_path)[:2]}
english_data_dict['agr_en_dev'].head()
english_data_dict['agr_en_train'].head()
english_data_dict['agr_en_train'].shape
len_comments = [len(i) for i in english_data_dict['agr_en_train']['comment'].tolist()]
english_data_dict['agr_en_train']['comment_length'] = len_comments
english_data_dict['agr_en_train'].head()

english_data_dict['agr_en_train']['comment_length'].describe()
