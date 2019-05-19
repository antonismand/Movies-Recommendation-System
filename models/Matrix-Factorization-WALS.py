import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix

from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE

from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import sparse_ops
from tensorflow.contrib.factorization.python.ops import factorization_ops
from tensorflow.contrib.factorization.python.ops import factorization_ops_test_utils

import tensorflow as tf
import pandas as pd
import numpy as np
import math
import os

np_matrix_to_tf_sparse = factorization_ops_test_utils.np_matrix_to_tf_sparse



def np_matrix_to_tf_sparse(np_matrix, i,
                           row_slices=None,
                           col_slices=None,
                           transpose=False,
                           shuffle=False):
  """Simple util to slice non-zero np matrix elements as tf.SparseTensor."""
  indices = np.nonzero(np_matrix)

  # for r in row_slices:
  #   print(r-i)

  # Only allow slices of whole rows or whole columns.
  assert not (row_slices is not None and col_slices is not None)

  if row_slices is not None:
    selected_ind = np.concatenate([np.where(indices[0] == (r-i))[0] for r in row_slices], 0)
    in_real = (indices[0][selected_ind], indices[1][selected_ind])
    indices = (indices[0][selected_ind]+i, indices[1][selected_ind])

  # if col_slices is not None:
  #   selected_ind = np.concatenate([np.where(indices[1] == c)[0] for c in col_slices], 0)
  #   indices = (indices[0][selected_ind], indices[1][selected_ind])
  #   in_real = indices

  if shuffle:
    shuffled_ind = [x for x in range(len(indices[0]))]
    random.shuffle(shuffled_ind)
    indices = (indices[0][shuffled_ind], indices[1][shuffled_ind])

  ind = (np.concatenate((np.expand_dims(indices[1], 1), np.expand_dims(indices[0], 1)), 1).astype(np.int64) if
         transpose else np.concatenate((np.expand_dims(indices[0], 1), np.expand_dims(indices[1], 1)), 1).astype(np.int64))

  val = np_matrix[in_real].astype(np.float32)
  shape = (np.array([max(indices[1]) + 1, max(indices[0]) + 1]).astype(np.int64)
           if transpose else np.array(
               [max(indices[0]) + 1, max(indices[1]) + 1]).astype(np.int64))
  return sparse_tensor.SparseTensor(ind, val, shape)






def get_MAE(output_row, output_col, actual):
    mae = 0
    for i in range(actual.data.shape[0]):
        row_pred = output_row[actual.row[i]]
        col_pred = output_col[actual.col[i]]
        mae += abs(actual.data[i] - np.dot(row_pred, col_pred))
    mae /= actual.data.shape[0]
    return mae

def get_RMSE(output_row, output_col, actual):
    mse = 0
    for i in range(actual.data.shape[0]):
        row_pred = output_row[actual.row[i]]
        col_pred = output_col[actual.col[i]]
        err = actual.data[i] - np.dot(row_pred, col_pred)
        mse += err * err
    mse /= actual.data.shape[0]
    rmse = math.sqrt(mse)
    return rmse

def evaluate_model(sess, train_sparse, test_sparse, row_factor, col_factor):

    train_rmse = get_RMSE(row_factor, col_factor, train_sparse)
    test_rmse = get_RMSE(row_factor, col_factor, test_sparse)    

    print('train RMSE: ', train_rmse)
    print('test RMSE: ', test_rmse)

    tf.logging.info('train RMSE = %f' % train_rmse)
    tf.logging.info('test RMSE = %f' % test_rmse)

    train_mae = get_MAE(row_factor, col_factor, train_sparse)
    test_mae = get_MAE(row_factor, col_factor, test_sparse)    

    print('train MAE: ', train_mae)
    print('test MAE: ', test_mae)

    tf.logging.info('train MAE = %f' % train_mae)
    tf.logging.info('test MAE = %f' % test_mae)



#########################################################################################

print("\nLoading Dataset....\n")


input_file = '../datasets/ml-latest/ratings.csv'
headers = ['userId', 'movieId', 'rating', 'timestamp']
header_row = None
ratings_df = pd.read_csv(input_file, sep=",", names=headers, header=header_row, skiprows = 1,
                         dtype={'userId': np.int32, 'movieId': np.int32, 
                                'rating': np.float32,'timestamp': np.int32,
                         })

# ratings_df = ratings_df.head(len(ratings_df)//5)

#########################################################################################


print("\nData Preprocessing....\n")

np_users = ratings_df.userId.values
np_items = ratings_df.movieId.values

unique_users = np.unique(np_users)
unique_items = np.unique(np_items)

n_users = unique_users.shape[0]
n_items = unique_items.shape[0]

print(n_users, n_items)

max_item = unique_items[-1]

# Reconstruct the ratings set's user/movie indices
np_users = ratings_df.userId.values
np_users[:] -= 1 # Make users zero-indexed

# Mapping unique items down to an array 0..n_items-1
z = np.zeros(max_item+1, dtype=int)
z[unique_items] = np.arange(n_items)
movies_map = z[np_items]

np_ratings = ratings_df.rating.values
# print(np_ratings.shape[0])
ratings = np.zeros((np_ratings.shape[0], 3), dtype=object)
ratings[:, 0] = np_users
ratings[:, 1] = movies_map
ratings[:, 2] = np_ratings

X_train, X_test = train_test_split(ratings, train_size=0.8)

# Ignoring timestamp
user_train, movie_train, rating_train = zip(*X_train)
train_sparse = coo_matrix((rating_train, (user_train, movie_train)), shape=(n_users, n_items))
# print(train_sparse)

user_test, movie_test, rating_test = zip(*X_test)
test_sparse = coo_matrix((rating_test, (user_test, movie_test)), shape=(n_users, n_items))
# print(test_sparse)



print("\nBuilding Model....\n")


# Default hyperparameters
DEFAULT_PARAMS = {
    'weights': True,
    'latent_factors': 5,
    'num_iters': 20,
    'regularization': 0.07,
    'unobs_weight': 0.01,
    'wt_type': 0,
    'feature_wt_factor': 130.0,
    'feature_wt_exp': 0.08,
    'delimiter': '\t'
}

# Parameters optimized with hypertuning for the MovieLens data set
OPTIMIZED_PARAMS = {
    'latent_factors': 34,
    'regularization': 9.83,
    'unobs_weight': 0.001,
    'feature_wt_factor': 189.8,
}

params = DEFAULT_PARAMS

# Create WALS model
row_wts = None
col_wts = None

num_rows = train_sparse.shape[0]
num_cols = train_sparse.shape[1]

sess = tf.Session()  #graph=input_tensor.graph)

model = factorization_ops.WALSModel(num_rows, num_cols, 
                                    n_components=params['latent_factors'],
                                    # num_row_shards=2,
                                    # num_col_shards=3,
                                    unobserved_weight=params['unobs_weight'],
                                    regularization=params['regularization'],
                                    row_weights=row_wts, 
                                    col_weights=col_wts)

print("\nPreparation for Training....\n")

with tf.Session() as sess:

    sess.run(model.initialize_op)
    sess.run(model.worker_init)

        
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    batch = 512
    X_train = train_sparse.tocsr()

    k_rows = num_rows // batch
    k_cols = num_cols // batch

    print(num_rows, k_rows)
    print(num_cols, k_cols)
    
    # ~~~~~~~~~ Cutting smaller Row Slices - mini batching ~~~~~~~~~ #
    input_scattered_rows = []
    for i in range(k_rows+1):

        to = (i+1)*batch
        if to > num_rows:
            to = num_rows

        x = X_train[i*batch : (i+1)*batch]
        x = x.toarray()
        slice = np_matrix_to_tf_sparse(x, i*batch, row_slices=np.arange(i*batch, to), transpose=False).eval()
        input_scattered_rows.append(slice)

    sp_feeder_row = array_ops.sparse_placeholder(dtypes.float32)
    (_, slice_row, unreg_loss_row, reg_row, _) = model.update_row_factors(sp_input=sp_feeder_row, transpose_input=False)
    factor_loss_row = unreg_loss_row + reg_row


    # ~~~~~~~~~ Cutting smaller Column Slices - mini batching ~~~~~~~~~ #
    input_scattered_cols = []
    for i in range(k_cols+1):

        to = (i+1)*batch
        if to > num_cols:
            to = num_cols

        x = X_train[:, i*batch : (i+1)*batch]
        x = np.transpose(x.toarray())

        slice = np_matrix_to_tf_sparse(x, i*batch, row_slices=np.arange(i*batch, to), transpose=False).eval()
        input_scattered_cols.append(slice)

    sp_feeder_col = array_ops.sparse_placeholder(dtypes.float32)
    (_, slice_col, unreg_loss_col, reg_col,_) = model.update_col_factors(sp_input=sp_feeder_col, transpose_input=True)
    factor_loss_col = unreg_loss_col + reg_col
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


    print("\nTraining....\n")
    num_iterations = 1
    for i in range(num_iterations):

        # ROW SWEEP
        #####################################################################################
        model.row_update_prep_gramian_op.run(session=sess)                  # Step 1
        model.initialize_row_update_op.run(session=sess)                    # Step 2
        for inp in input_scattered_rows:
            feed_dict = {sp_feeder_row: inp}
            slice_row.run(session=sess, feed_dict=feed_dict)                # Step 3
        row_factors = [x.eval() for x in model.row_factors]
        # print(row_factors)
        # print(len(row_factors))
        #####################################################################################

        # COLUMN SWEEP
        #####################################################################################
        model.col_update_prep_gramian_op.run(session=sess)                  # Step 1
        model.initialize_col_update_op.run(session=sess)                    # Step 2
        for inp in input_scattered_cols:
            feed_dict = {sp_feeder_col: inp}
            slice_col.run(session=sess, feed_dict=feed_dict)                # Step 3
        col_factors = [x.eval() for x in model.col_factors]
        # print(col_factors)
        # print(len(col_factors))
        #####################################################################################

        if i%4 == 0:
            print("\nEvaluating..: ", i, "/", num_iterations)
            evaluate_model(sess, train_sparse, test_sparse, row_factors[0], col_factors[0])

# if num_iterations%3 == 0:
print("\nEvaluating..: ", num_iterations, "/", num_iterations)
evaluate_model(sess, train_sparse, test_sparse, row_factors[0], col_factors[0])



# print("\Saving Model....\n")

# # Evaluate output factor matrices
# output_row = row_factor.eval(session=sess)
# output_col = col_factor.eval(session=sess)

# model_dir = os.path.join("WALS", 'model')
# os.makedirs(model_dir)
# np.save(os.path.join(model_dir, 'user'), np_users)
# np.save(os.path.join(model_dir, 'movie'), movies_map)
# np.save(os.path.join(model_dir, 'row'), output_row)
# np.save(os.path.join(model_dir, 'col'), output_col)

