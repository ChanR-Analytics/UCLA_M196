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

    def frame_process(self, results):
        frame_dict_list = []
        school_count = 0
        while school_count < len(results):
            names = []
            latitudes = []
            longitudes = []
            total_user_ratings = []
            ratings = []

            for i in range(len(results[school_count]['results'])):
                name = results[school_count]['results'][i]['name']
                latitude = results[school_count]['results'][i]['geometry']['location']['lat']
                longitude = results[school_count]['results'][i]['geometry']['location']['lng']
                total_user_rating = results[school_count]['results'][i]['user_ratings_total']
                rating = results[school_count]['results'][i]['rating']
                names.append(name)
                latitudes.append(latitude)
                longitudes.append(longitude)
                total_user_ratings.append(total_user_rating)
                ratings.append(rating)

            frame_dict = {'names': names, 'latitudes': latitudes, 'longitudes': longitudes, 'total_user_ratings': total_user_ratings, 'rating': ratings}
            frame_dict_list.append(frame_dict)
            school_count += 1

        frame_list = [pd.DataFrame.from_dict(frame_dict) for frame_dict in frame_dict_list]
        school_names = self.df['school_name'].tolist()

        return {school: frame_list[i] for i, school in enumerate(school_names)}

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
        return distance_dict

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
                    distance = 'NaN'
                    duration = 'NaN'
                else:
                    distance = element['distance']['text']
                    duration = element['duration']['text']
                distances.append(distance)
                durations.append(duration)

            frame_dict[school] = pd.DataFrame.from_dict({'distances': distances, 'durations': durations})

        return frame_dict
