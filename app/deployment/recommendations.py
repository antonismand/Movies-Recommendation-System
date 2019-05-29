from __future__ import print_function
import logging
import numpy as np
import pandas as pd
import sys
import os

model = "MF-WALS"

LOCAL_MODEL_PATH = './saved-model/' + model

ROW_MODEL_FILE = 'row.npy'
COL_MODEL_FILE = 'col.npy'
USER_MODEL_FILE = 'user.npy'
ITEM_MODEL_FILE = 'movie.npy'
MOVIES_ID_FILE = 'movie_ids.npy'
#USER_ITEM_DATA_FILE = "./dataset/u.data"
USER_ITEM_DATA_FILE = "./dataset/u1.test"
ITEM_DATA_FILE = "./dataset/u.item"
POSTERS_DATA_FILE = "./dataset/movie_posters/movie_poster.csv"


class Recommendations(object):

    def __init__(self, local_model_path=LOCAL_MODEL_PATH):
        self._load_model(local_model_path)
        self._load_data()
        
        
    def _load_data(self):
        
        self.movies_df = pd.read_csv(ITEM_DATA_FILE, sep='|', header=None)
        self.movies_df = self.movies_df.iloc[:, 0:2]
        self.movies_df.columns = ["movie_id", "title"]
        self.movies = self.movies_df.set_index('movie_id').T.to_dict('list')

        self.posters_df = pd.read_csv(POSTERS_DATA_FILE, sep=',', header=None)
        self.posters_df = self.posters_df.iloc[:, 0:2]
        self.posters_df.columns = ["movie_id", "pic_url"]
        self.posters = self.posters_df.set_index('movie_id').T.to_dict('list')
        

    def _load_model(self, local_model_path):

        # load npy arrays for user/item factors and user/item maps
        self.user_factor = np.load(os.path.join(local_model_path, ROW_MODEL_FILE),allow_pickle=True)
        self.item_factor = np.load(os.path.join(local_model_path, COL_MODEL_FILE),allow_pickle=True)
        self.user_map = np.load(os.path.join(local_model_path, USER_MODEL_FILE),allow_pickle=True)
        self.item_map = np.load(os.path.join(local_model_path, ITEM_MODEL_FILE),allow_pickle=True)
        self.movies_id_map = np.load(os.path.join(local_model_path, MOVIES_ID_FILE),allow_pickle=True)

        logging.info('Finished loading arrays.')

        # load user_item history into pandas dataframe
        headers = ['userId', 'movieId', 'rating', 'timestamp']
        views_df = pd.read_csv(USER_ITEM_DATA_FILE, sep='\s+', names=headers, header=None,
                              dtype={'userId': np.int32, 'movieId': np.int32,
                                    'rating': np.float32,'timestamp': np.int32
                             })
        self.user_items = views_df.groupby('userId')

        logging.info('Finished loading model.')


    def get_recommendations(self, user_id, num_recs):

        data = []
        user_idx = user_id - 1
        # get already viewed items from views dataframe
        already_rated = self.user_items.get_group(user_id).movieId
        already_rated_idx = [np.searchsorted(self.item_map, i) for i in already_rated]
        # generate list of recommended movie indexes from model
        recommendations = self.generate_recommendations(user_idx, already_rated_idx,
                                             self.user_factor, self.item_factor, num_recs)
        for i in recommendations:
            if self.item_map[i] in self.movies:
                title = self.movies[self.item_map[i]]
            else:
                title = "E"
            if self.item_map[i] in self.posters:
                pic_url = self.posters[self.item_map[i]]
            else:
                pic_url = "E"

            data += [(title, pic_url)]

#         print(recommendations, file=sys.stderr)
#         print(data, file=sys.stderr)

        return  data


    def get_prediction(self, user_id, movie_id):
        
        rating = None
        user_idx = user_id - 1
        # movie_idx = self.movies_id_map[movie_id]
        rating = self.generate_prediction(user_idx, movie_id-1, self.user_factor, self.item_factor)

        # get already viewed items from views dataframe
        user_group = self.user_items.get_group(user_id)
        past_movies = user_group.movieId
        past_ratings = user_group.rating
        
        actual_rating = 0
        for (m, r) in zip(past_movies, past_ratings):
            if m == movie_id:
                actual_rating = r
                break

        return rating, actual_rating


    def generate_recommendations(self, user_idx, user_rated, row_factor, col_factor, k):

        # bounds checking for args
        assert (row_factor.shape[0] - len(user_rated)) >= k

        # retrieve user factor
        user_f = row_factor[user_idx]

        # dot product of item factors with user factor gives predicted ratings
        pred_ratings = col_factor.dot(user_f)

        # find candidate recommended item indexes sorted by predicted rating
        k_r = k + len(user_rated)
        candidate_items = np.argsort(pred_ratings)[-k_r:]

        # remove previously rated items and take top k
        recommended_items = [i for i in candidate_items if i not in user_rated]
        recommended_items = recommended_items[-k:]

        # flip to sort highest rated first
        recommended_items.reverse()

        return recommended_items


    def generate_prediction(self, user_idx, movie_idx, row_factor, col_factor):

        # retrieve user factor
        user_f = row_factor[user_idx]

        # dot product of item factors with user factor gives predicted ratings
        pred_ratings = col_factor.dot(user_f)
        rating = pred_ratings[movie_idx]
        rating = round(rating, 1)
        return rating
