import numpy as np
import pandas as pd
from os import getcwd, listdir

data_path = getcwd() + "/geospatial_project/data/geo_results"
df_dict = {i[:-4] : pd.read_csv(f"{data_path}/{i}") for i in listdir(data_path)}
