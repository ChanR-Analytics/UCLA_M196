import numpy as np
import pandas as pd
import googlemaps
from os import listdir

class nearest_restaurants:
    def __init__(self, data_path):
        self.data_path = data_path

    def initialize(self):
        self.key = "AIzaSyBHWiHNgsyEL8IzkG42rcZYmqzjIXXHswE"
        self.gmaps = googlemaps.Client(key = self.key)
        return self.gmaps

    def view_schools(self):
        self.df = pd.read_csv(f"{data_path}/{listdir(data_path)[0]}")
        return self.df

    def make_coordinates(self):
        

    def search_results(self, query, radius, now):
        self.query = query
        self.radius = radius
        self.now = now

        self.results = []

        for name in self.school_names:
            self.result = self.gmaps.places(self.query, )
