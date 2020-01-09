from os import getcwd, listdir
from geospatial_project.src.scripts.nearest_restaurants_schools_oop import nearest_restaurants

school_path = getcwd() + "/geospatial_project/data/csv"

nr = nearest_restaurants(school_path)
nr.view_schools()
nr.make_coordinates()
nr_results = nr.search_results(query='restaurants', radius=10, now=True)

nr_frame_dict = nr.frame_process(nr_results)

nr_frame_dict['Arcadia High School'].head()

haversine_results = nr.haversine_distance(nr_frame_dict)

haversine_results['Arcadia High School']
