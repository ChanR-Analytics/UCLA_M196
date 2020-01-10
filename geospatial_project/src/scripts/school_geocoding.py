import googlemaps
import numpy as np
import pandas as pd
from os import getcwd, listdir
from getpass import getpass

# Transform the School Names into a List of Strings
txt_path = getcwd() + "/geospatial_project/data/txt"
listdir(txt_path)

school_names = []
with open(f"{txt_path}/{listdir(txt_path)[1]}", "r") as my_file:
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

# Configuring Google Maps Client
gmaps = googlemaps.Client(key= getpass("Input your API key"))

geocode_results = []

for name in school_names:
    geocode_result = gmaps.geocode(name)
    geocode_results.append(geocode_result)

geocode_results[0][0]['geometry']['location']

coordinates = []

count = 0
while count < len(school_names):
    lat = geocode_results[count][0]['geometry']['location']['lat']
    lng = geocode_results[count][0]['geometry']['location']['lng']
    coordPair = (school_names[count], lat, lng)
    coordinates.append(coordPair)
    count += 1

lat_vals = []
long_vals = []

for coord in coordinates:
    lat_vals.append(coord[1])
    long_vals.append(coord[2])

coord_df = pd.DataFrame.from_dict({'school_name': school_names, 'latitude': lat_vals, 'longitude': long_vals})
csv_path = getcwd() + "/geospatial_project/data/csv"
coord_df.to_csv(f"{csv_path}/school_coordinates.csv", index=False)
