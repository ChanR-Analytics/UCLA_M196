import numpy as np
import pandas as pd
from geospatial_project.src.scripts.nearest_restaurants_schools_oop import nearest_restaurants
from os import getcwd, listdir

# Getting Schools
school_path = getcwd() + "/geospatial_project/data/csv"
nr = nearest_restaurants(school_path)
school_coords = nr.make_coordinates()
school_coord_dict = {school: school_coords[i] for i, school in enumerate(nr.df['school_name'].tolist())}

# Getting Seach Results
nr_results = nr.search_results(query="restaurants", radius=10, now=True)
nr_frame_dict = nr.frame_process(nr_results)

# Using Distance Matrix API to Get Various Route Distances
google_dist_dict = {}

gmaps = nr.gmaps

result_dict = {}
for school in list(nr_frame_dict.keys()):
    restaurant_lat = nr_frame_dict[school]['latitudes'].tolist()
    restaurant_long = nr_frame_dict[school]['longitudes'].tolist()
    restaurant_coordinates = list(zip(restaurant_lat, restaurant_long))
    result = gmaps.distance_matrix(origins=school_coord_dict[school], destinations=restaurant_coordinates, mode='walking')
    result_dict[school] = result
