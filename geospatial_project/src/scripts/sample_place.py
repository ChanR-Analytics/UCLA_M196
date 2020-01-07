import googlemaps
import numpy as np
import pandas as pd
from getpass import getpass
from os import getcwd,listdir


# Getting Coordinates from School Names
coord_path = getcwd() + "/geospatial_project/data/csv"

coord_df = pd.read_csv(f"{coord_path}/{listdir(coord_path)[0]}")

lat_vals = coord_df['latitude'].tolist()
long_vals = coord_df['longitude'].tolist()

coordinates = list(zip(lat_vals, long_vals))

print(coordinates)

gmaps = googlemaps.Client(key=getpass("Type in your API key: "))

place_result = gmaps.places('restaurant', radius=10, location=coordinates[1])
place_result.keys()
place_result['results'][0].keys()
place_result['results'][0]['name']
place_result['results'][0]['geometry']
place_result['results'][0]['user_ratings_total']
place_result['results'][0]['photos']
place_result['results'][0]['photos'][0]['html_attributions']
place_result['results'][0]['photos'][0]['photo_reference']
len(place_result['results'])
