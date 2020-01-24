import numpy as np
import pandas as pd
from os import getcwd, listdir

data_path = getcwd() + "/geospatial_project/data/geo_results"
df_dict = {i[:-4] : pd.read_csv(f"{data_path}/{i}") for i in listdir(data_path)}

df_results = list(df_dict.values())

new_df = pd.concat(df_results, axis=0, sort=True)
new_df.reset_index(inplace=True)
new_df.drop("index", axis=1, inplace=True)
new_df.to_csv(f"{data_path}/merged_geo_results.csv", index=False)
