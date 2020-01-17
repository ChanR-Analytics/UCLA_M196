def frame_process(self, result_dict):
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
