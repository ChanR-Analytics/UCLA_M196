import numpy as np
import pandas as pd
from geospatial_project.src.scripts.nearest_restaurants_schools_oop import nearest_restaurants
from os import getcwd, listdir
from IPython.display import display
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
    result = gmaps.distance_matrix(origins=school_coord_dict[school], destinations=restaurant_coordinates, mode='driving')
    result_dict[school] = result

distance_frame_dict = {}
duration_frame_dict = {}

len(result_dict['Northwood High School']['rows'][0]['elements'])
result_dict['Northwood High School']
school_names = list(nr_frame_dict.keys())
school_names.remove('Northwood High School')
school_names

frame_dict = {}

for school in list(nr_frame_dict.keys()):
    distances = []
    durations = []
    for element in result_dict[school]['rows'][0]['elements']:
        if element['status'] == 'ZERO_RESULTS':
            distance = 'NaN'
            duration = 'NaN'
        else:
            distance = element['distance']['text']
            duration = element['duration']['text']
        distances.append(distance)
        durations.append(duration)
    frame_dict[school] = pd.DataFrame.from_dict({'distances': distances, 'durations': durations})


school_names = list(frame_dict.keys())

for school in school_names:
    display(frame_dict[school])
