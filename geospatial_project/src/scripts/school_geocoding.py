import googlemaps
import numpy as np
import pandas as pd
from os import getcwd, listdir

# Transform the School Names into a List of Strings
txt_path = getcwd() + "/geospatial_project/data/txt"
listdir(txt_path)

school_names = []
with open(f"{txt_path}/{listdir(txt_path)[0]}", "r") as my_file:
    for line in my_file.readlines():
        line = line.replace("\n", "")
        school_names.append(line)

print(school_names)

new_school_names = []

for name in school_names[1:]:
    name_list = name.split(" ")
    name_list.insert(1, 'High')
    new_school_names.append(" ".join(name_list))

school_names = [school_names[0]] + new_school_names

print(school_names)
