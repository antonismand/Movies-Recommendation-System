import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
import pickle


# Computes mae in batches so that we don't have memory issue in big dataset
def compute_MAE(sess, train_sparse, output_layer):

	loss_mae = 0
	bsize=100
	tot_users = train_sparse.shape[0]
	train_sparse = train_sparse.tocsr()
	# print(tot_users)

	for i in range(int(tot_users/bsize)+1):

		to = (i+1)*bsize
		if to > tot_users:
			to = tot_users
			
		epoch_x = train_sparse[ i*bsize : to ]
		epoch_x = epoch_x.toarray()

		output_train = sess.run(output_layer, feed_dict={input_layer:epoch_x})
		loss_mae += np.sum(abs(output_train - epoch_x))

	mae = loss_mae / (train_sparse.shape[0]*train_sparse.shape[1])
	return mae


def compute_RMSE(sess, train_sparse, output_layer):

	loss_rmse = 0
	bsize=100
	tot_users = train_sparse.shape[0]
	train_sparse = train_sparse.tocsr()
	# print(tot_users)

	for i in range(int(tot_users/bsize)+1):

		to = (i+1)*bsize
		if to > tot_users:
			to = tot_users
			
		epoch_x = train_sparse[ i*bsize : to ]
		epoch_x = epoch_x.toarray()

		output_train = sess.run(output_layer, feed_dict={input_layer:epoch_x})
		loss_rmse += np.sum(np.square(output_train - epoch_x))

	rmse = loss_rmse / (train_sparse.shape[0]*train_sparse.shape[1])
	return rmse


def save_weights(sess, hidden_vals, output_vals):

	v1 = sess.run(hidden_vals)
	f = open("weights-hidden.pkl","wb")
	pickle.dump(v1,f)
	f.close()

	v2 = sess.run(output_vals)
	f = open("weights-output.pkl","wb")
	pickle.dump(v2,f)
	f.close()

###############################################################################################
# Reading the ratings data
ratings_df = pd.read_csv('../datasets/ml-latest/ratings.csv', sep=",")

# Making the dataset a little bit smaller due to lack of memory resources
ratings_df = ratings_df.head(len(ratings_df))
# print(ratings_df)

# Preprocessing
np_users = ratings_df.userId.values
np_items = ratings_df.movieId.values

unique_users = np.unique(np_users)
unique_items = np.unique(np_items)

n_users = unique_users.shape[0]
n_items = unique_items.shape[0]

# print(n_users)
# print(n_items)

max_item = unique_items[-1]

# Reconstruct the ratings set's user/movie indices
np_users = ratings_df.userId.values
np_users[:] -= 1 # Make users zero-indexed
# print(np_users)

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
# print(X_train)

# Ignoring timestamp
user_train, movie_train, rating_train = zip(*X_train)
train_sparse = coo_matrix((rating_train, (user_train, movie_train)), shape=(n_users, n_items))
# print(train_sparse.shape)

user_test, movie_test, rating_test = zip(*X_test)
test_sparse = coo_matrix((rating_test, (user_test, movie_test)), shape=(n_users, n_items))
# print(test_sparse.shape)


# Deciding how many nodes each layer should have - Depending on the dataset's size
movies_size = 53889 #28267 #9724 #28267 #58099
n_nodes_inpl = movies_size
n_nodes_hl1  = 256
n_nodes_outl = movies_size

# with tf.device('/cpu:0'):

# first hidden layer has 15159*256 weights and 256 biases
hidden_1_layer_vals = {'weights':tf.Variable(tf.random_normal([n_nodes_inpl+1, n_nodes_hl1]))}
output_layer_vals = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1+1, n_nodes_outl]))}

input_layer = tf.placeholder('float', [None, movies_size])

input_layer_const = tf.fill([tf.shape(input_layer)[0], 1], 1.0)
input_layer_concat = tf.concat([input_layer, input_layer_const], 1)

layer_1 = tf.nn.sigmoid(tf.matmul(input_layer_concat, hidden_1_layer_vals['weights']))

layer1_const = tf.fill( [tf.shape(layer_1)[0], 1], 1.0)
layer_concat =  tf.concat([layer_1, layer1_const], 1)

output_layer = tf.matmul(layer_concat, output_layer_vals['weights'])

output_true = tf.placeholder('float', [None, movies_size])
meansq = tf.reduce_mean(tf.square(output_layer - output_true))

learn_rate = 0.1   # learning rate
optimizer = tf.train.AdagradOptimizer(learn_rate).minimize(meansq)

# initializing variables and starting the session
init = tf.global_variables_initializer()

config=tf.ConfigProto(log_device_placement=True)
sess = tf.Session(config=config)

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
writer.close()
sess.run(init)

batch_size = 100
hm_epochs = 1
tot_users = train_sparse.shape[0]
# print(tot_users)

train_sparse = train_sparse.tocsr()
for epoch in range(hm_epochs):
	epoch_loss = 0
	for i in range(int(tot_users/batch_size)):
		epoch_x = train_sparse[ i*batch_size : (i+1)*batch_size ]
		epoch_x = epoch_x.toarray()
		_, c = sess.run([optimizer, meansq],feed_dict={input_layer: epoch_x, output_true: epoch_x})
		epoch_loss += c
  
	print((epoch+1)%10)
	if (epoch+1) % 10 == 0:
		print('MAE train', compute_MAE(sess, train_sparse, output_layer))
		print('MSE train', np.sqrt(compute_RMSE(sess, train_sparse, output_layer)))

print('MAE train', compute_MAE(sess, train_sparse, output_layer))
print('MSE train', np.sqrt(compute_RMSE(sess, train_sparse, output_layer)))
print('Epoch', epoch, '/', hm_epochs, 'loss:',epoch_loss)

###############################################################################################