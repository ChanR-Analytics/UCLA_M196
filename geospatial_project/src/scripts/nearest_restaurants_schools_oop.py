import numpy as np
import pandas as pd
import googlemaps
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

    self.coordinates = self.make_coordinates() 

    def search_results(self, query, radius, now):
        self.query = query
        self.radius = radius
        self.now = now

        self.results = []
        for coordinate in self.coordinates:
            self.result = self.gmaps.places(self.query, location=coordinate, radius=self.radius, open_now=self.now)
            self.results.append(self.result)

        return self.results

    def frame_process(self):
        return self.df
