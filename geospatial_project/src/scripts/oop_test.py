from os import getcwd, listdir
from geospatial_project.src.scripts.nearest_restaurants_schools_oop import nearest_restaurants

school_path = getcwd() + "/geospatial_project/data/csv"

nr = nearest_restaurants(school_path)
nr.view_schools()
nr.make_coordinates()
nr.search_results(query='restaurants', radius=10, now=True)
