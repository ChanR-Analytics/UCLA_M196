import numpy as np
import pandas as pd
import googlemaps
from haversine import haversine
from os import listdir



class nearest_restaurants:
    def __init__(self, data_path):
        self.data_path = data_path
        self.key = "AIzaSyBHWiHNgsyEL8IzkG42rcZYmqzjIXXHswE"
        self.gmaps = googlemaps.Client(key = self.key)
        self.df = pd.read_csv(f"{self.data_path}/{listdir(data_path)[0]}")

    def view_schools(self):
        return self.df

    def make_coordinates(self):
        self.lat = self.df['latitude'].tolist()
        self.long = self.df['longitude'].tolist()
        coordinates = list(zip(self.lat, self.long))
        return coordinates

    def search_results(self, query, radius, now):
        self.query = query
        self.radius = radius
        self.now = now
        self.coordinates = self.make_coordinates()

        self.results = []
        for coordinate in self.coordinates:
            self.result = self.gmaps.places(self.query, location=coordinate, radius=self.radius, open_now=self.now)
            self.results.append(self.result)

        return self.results


    def frame_process(self, result_dict):
        format_result_dict = {}

        for school in list(result_dict.keys()):
            names = []
            latitudes = []
            longitudes = []
            total_user_ratings = []
            ratings = []
            price_levels = []
            for result_structure in result_dict[school]:
                for result in result_structure['results']:
                    name = result['name']
                    latitude = result['geometry']['location']['lat']
                    longitude = result['geometry']['location']['lng']
                    user_rating = result['user_ratings_total']
                    rating = result['rating']
                    if 'price_level' in result.keys():
                        price_level = result['price_level']
                    else:
                        price_level = float('nan')
                    names.append(name)
                    latitudes.append(latitude)
                    longitudes.append(longitude)
                    total_user_ratings.append(user_rating)
                    ratings.append(rating)
                    price_levels.append(price_level)
                format_result_dict[school] = {'names': names, 'latitudes': latitudes, 'longitudes': longitudes, 'total_user_ratings': total_user_ratings, 'ratings': ratings, 'price_levels': price_levels}

        format_result_df_dict = {school: pd.DataFrame.from_dict(format_result_dict[school]) for school in format_result_dict.keys()}
        return format_result_df_dict



    def haversine_distance(self, school_results_dict, metric):
        distance_dict = {}
        school_coordinates = self.make_coordinates()
        school_coord_dict = {school: school_coordinates[i] for i, school in enumerate(self.df['school_name'].tolist())}

        for school in list(school_results_dict.keys()):
            distance_list = []
            restaurant_lat = school_results_dict[school]['latitudes'].tolist()
            restaurant_long = school_results_dict[school]['longitudes'].tolist()
            restaurant_coordinates = list(zip(restaurant_lat, restaurant_long))

            for restaurant_coordinate in restaurant_coordinates:
                dist = haversine(school_coord_dict[school], restaurant_coordinate, unit=metric)
                distance_list.append(dist)
            distance_dict[school] = distance_list

        haversine_df_dict = {school: pd.DataFrame.from_dict({'haversine_distance': distance_dict[school]}) for school in distance_dict.keys()}
        return haversine_df_dict

    def google_distance(self, frame_dict, transporation_mode):

        school_coordinates = self.make_coordinates()
        school_coordinate_dict = {school : school_coordinates[i] for i, school in enumerate(self.df['school_name'].tolist())}
        result_dict = {}

        for school in list(frame_dict.keys()):
            latitudes = frame_dict[school]['latitudes'].tolist()
            longitudes = frame_dict[school]['longitudes'].tolist()
            restaurant_coordinates = list(zip(latitudes, longitudes))
            result = self.gmaps.distance_matrix(origins=school_coordinate_dict[school], destinations=restaurant_coordinates, mode=transporation_mode)
            result_dict[school] = result

        frame_dict = {}

        for school in list(result_dict.keys()):
            results = result_dict[school]['rows'][0]['elements']
            distances = []
            durations = []

            for element in results:

                if element['status'] == 'ZERO_RESULTS':
                    distance = float('nan')
                    duration = float('nan')
                else:
                    distance = element['distance']['text']
                    duration = element['duration']['text']
                distances.append(distance)
                durations.append(duration)

            frame_dict[school] = pd.DataFrame.from_dict({'distance_from_school': distances, f'{transporation_mode} duration': durations})

        return frame_dict
