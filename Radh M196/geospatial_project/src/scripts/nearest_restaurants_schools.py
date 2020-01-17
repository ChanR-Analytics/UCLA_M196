import numpy as np
import pandas as pd
import googlemaps
from os import getcwd, listdir
from getpass import getpass

# Getting Coordinates of Schools
coord_path = getcwd() + "/geospatial_project/data/csv"
coord_df = pd.read_csv(f"{coord_path}/{listdir(coord_path)[0]}")
coordinates = list(zip(coord_df['latitude'].tolist(), coord_df['longitude'].tolist()))

# Establishing Google Maps Client with API Key
gmaps = googlemaps.Client(key=getpass("Type in your API key: "))

# Search Restaurants Within a 10 Meter Radius from Each School
nearest_restaurant_results = []

for coord in coordinates:
    search_result = gmaps.places('restaurants', location=coord, radius=10, open_now=True)
    nearest_restaurant_results.append(search_result)


# Creating a Data Frame of Results for Each School
frame_dict_list = []
school_count = 0
while school_count < len(nearest_restaurant_results):
    names = []
    latitudes = []
    longitudes = []
    total_user_ratings = []
    ratings = []
    for i in range(len(nearest_restaurant_results[school_count]['results'])):
         name = nearest_restaurant_results[school_count]['results'][i]['name']
         latitude = nearest_restaurant_results[school_count]['results'][i]['geometry']['location']['lat']
         longitude = nearest_restaurant_results[school_count]['results'][i]['geometry']['location']['lng']
         total_user_rating = nearest_restaurant_results[school_count]['results'][i]['user_ratings_total']
         rating = nearest_restaurant_results[school_count]['results'][i]['rating']
         names.append(name)
         latitudes.append(latitude)
         longitudes.append(longitude)
         total_user_ratings.append(total_user_rating)
         ratings.append(rating)

    frame_dict = {'names': names, 'latitudes': latitudes, 'longitudes': longitudes, 'total_user_ratings': total_user_ratings, 'rating': ratings}
    frame_dict_list.append(frame_dict)
    school_count += 1

frame_list = [pd.DataFrame.from_dict(frame_dict) for frame_dict in frame_dict_list]

school_names = coord_df['school_name'].tolist()

school_results_dict = {school: frame_list[i] for i, school in enumerate(school_names)}
