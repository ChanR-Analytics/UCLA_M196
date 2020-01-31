from time import sleep
from os import getcwd, listdir
from geospatial_project.src.scripts.nearest_restaurants_schools_oop import nearest_restaurants
from IPython.display import display

school_path = getcwd() + "/geospatial_project/data/csv/school_coordinates.csv"

nr = nearest_restaurants(school_path)
nr.view_schools()
nr.make_coordinates()

nr_results = nr.search_results(query='restaurants', radius=10, now=False)

nr_results[0]


# Rather than Putting Page Result 2 and 3 in the Class, add them from the instantiated object
count = 0

school_names = nr.df['school_name'].tolist()

result_dict = {}
while count < len(nr_results):
    result = nr_results[count]
    sleep(3)
    result2 = nr.gmaps.places(query='restaurants', page_token=result['next_page_token'])
    sleep(3)
    result3 = nr.gmaps.places(query='restaurants', page_token=result2['next_page_token'])
    result_dict[school_names[count]] = [result, result2, result3]
    count += 1

nr_frame_dict = nr.frame_process(result_dict)

nr_frame_dict['Granada Hills Charter High School'].head()

haversine_results = nr.haversine_distance(nr_frame_dict, 'km')

google_results = nr.google_distance(nr_frame_dict, transporation_mode='walking')
google_results.keys()
for school in google_results.keys():
    display(google_results[school].dropna().reset_index().drop('index', axis=1))


merged_df_results = nr.merge_frames(nr_frame_dict, haversine_results, google_results)

merged_df_results['Arcadia High School']
