import numpy as np
import pandas as pd
import googlemaps
from os import getcwd, listdir

data_path = getcwd() + "/geospatial_project/data/csv"

df = pd.read_csv(f"{data_path}/{listdir(data_path)[0]}")

school_names = df['school_name'].tolist()

latitudes = df['latitude'].tolist()
longitudes = df['longitude'].tolist()

coordinates = list(zip(latitudes, longitudes))

coord_dict = {school: coordinates[i] for i, school in enumerate(school_names)}

gmaps = googlemaps.Client(key="AIzaSyBHWiHNgsyEL8IzkG42rcZYmqzjIXXHswE")

result = gmaps.places('restaurants', location=coord_dict['Arcadia High School'], open_now=False)
result_page_2 = gmaps.places('restaurants', page_token=result['next_page_token'])
result_page_3 = gmaps.places('restaurants', page_token=result_page_2['next_page_token'])

result
result_page_2
result_page_3
