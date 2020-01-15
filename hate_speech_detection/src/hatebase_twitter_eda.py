import numpy as np
import pandas as pd
from os import getcwd, listdir

# HatebaseTwitter Data Path
data_path = getcwd() + "/hate_speech_detection/data/HatebaseTwitter"
listdir(data_path)
df = pd.read_csv(f"{data_path}/{listdir(data_path)[0]}")
df.shape
df.drop(list(df.columns)[0], axis=1, inplace=True)
df.head()
df.to_csv(f"{data_path}/final_hatebase_twitter.csv", index=False)

df.head()

class_values = df['class'].value_counts().to_dict()
class_values
class_values.keys()
labels = ['hateful', 'offensive', 'neither']

new_class_vals = {labels[key]: class_values[key] for key in class_values.keys()}
new_class_vals

new_class_val_props = {key : f"{round(new_class_vals[key] / sum([i for i in new_class_vals.values()]) , 2)*100}%" for key in new_class_vals.keys()}
new_class_val_props
