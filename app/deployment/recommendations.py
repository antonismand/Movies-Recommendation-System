import logging
import numpy as np
import os
import pandas as pd

model = "MF-WALS"

LOCAL_MODEL_PATH = './saved-model/' + model

ROW_MODEL_FILE = 'row.npy'
COL_MODEL_FILE = 'col.npy'
USER_MODEL_FILE = 'user.npy'
ITEM_MODEL_FILE = 'item.npy'

USER_ITEM_DATA_FILE = "./dataset/ratings.csv"

class Recommendations(object):

    def __init__(self, local_model_path=LOCAL_MODEL_PATH):
        self._load_model(local_model_path)
        print("Hey")

    def _load_model(self, local_model_path):

        # load npy arrays for user/item factors and user/item maps
#         self.user_factor = np.load(os.path.join(local_model_path, ROW_MODEL_FILE),allow_pickle=True)
#         self.item_factor = np.load(os.path.join(local_model_path, COL_MODEL_FILE),allow_pickle=True)
#         self.user_map = np.load(os.path.join(local_model_path, USER_MODEL_FILE),allow_pickle=True)
#         self.item_map = np.load(os.path.join(local_model_path, ITEM_MODEL_FILE),allow_pickle=True)

        logging.info('Finished loading arrays.')

        # load user_item history into pandas dataframe
        headers = ['userId', 'movieId', 'rating', 'timestamp']
        views_df = pd.read_csv(USER_ITEM_DATA_FILE, sep=',', 
                               names=headers, header=None, skiprows = 1)
        self.user_items = views_df.groupby('userId')

        logging.info('Finished loading model.')

        
    def get_recommendations(self, user_id, num_recs):

        movie_recommendations = None
        user_idx = user_id - 1
        if user_idx:
            # get already viewed items from views dataframe
            already_rated = self.user_items.get_group(user_id).movieId
            already_rated_idx = [np.searchsorted(self.item_map, i) for i in already_rated]
            # generate list of recommended movie indexes from model
            recommendations = generate_recommendations(user_idx, already_rated_idx,
                                                 self.user_factor, self.item_factor, num_recs)
        return  [self.item_map[i] for i in recommendations]


    def get_prediction(self, user_id, movie_id):

        rating = None
        user_idx = user_id - 1
        if user_idx:
            movie_idx = np.searchsorted(self.item_map, movie_id)
            rating = generate_rating(user_idx, movie_idx, self.user_factor, self.item_factor)

        return rating


def generate_recommendations(user_idx, user_rated, row_factor, col_factor, k):

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


def generate_predictions(user_idx, movie_idx, row_factor, col_factor):

    
    return None
#     # bounds checking for args
#     assert (row_factor.shape[0] - len(user_rated)) >= k

#     # retrieve user factor
#     user_f = row_factor[user_idx]

#     # dot product of item factors with user factor gives predicted ratings
#     pred_ratings = col_factor.dot(user_f)

#     # find candidate recommended item indexes sorted by predicted rating
#     k_r = k + len(user_rated)
#     candidate_items = np.argsort(pred_ratings)[-k_r:]

#     # remove previously rated items and take top k
#     recommended_items = [i for i in candidate_items if i not in user_rated]
#     recommended_items = recommended_items[-k:]

#     # flip to sort highest rated first
#     recommended_items.reverse()

#     return recommended_items