import json
import nltk
import itertools

import numpy as np 
import tensorflow as tf 

# Step 1: Read in the data and Tokenize it! 
# "Make America Great Again!" => ["Make", "America", "Great", "Again", "!""] (tokenized form)
# 
# [0, 17, 5, 2, 20, 50] => "<s> Make America Great Again!" (Input Example)
# [17, 5, 2, 20, 50, 1] => "Make America Great Again! </s>" (Label) 

# PREPROCESSING VARIABLES
vocab_size = 16000
unknown_token = "<unk>"
start_token = "<s>"
end_token = "</s>"

def preprocess():
	tokenized_tweet_lst = []
	with open('trump_tweets.txt', 'r') as f:
		for tweet in f:
			tokenized_tweet = nltk.tokenize.TweetTokenizer().tokenize(tweet.lower())
			tokenized_tweet = [start_token] + tokenized_tweet + [end_token]
			tokenized_tweet_lst.append(tokenized_tweet)

	# vocabularly of the most common 16000 "twits" 
	word_freq = nltk.FreqDist((itertools.chain(*tokenized_tweet_lst)))
	vocab = word_freq.most_common(vocab_size)

	# map a numerical representation to a twit
	index_to_word = {(index+1):word[0] for index,word in enumerate(vocab)}
	index_to_word[vocab_size] = unknown_token

	# map a twit to a numerical representation 
	word_to_index = {word[0]:(index+1) for index,word in enumerate(vocab)}
	word_to_index[unknown_token] = vocab_size
	end_token_idx = word_to_index[end_token]
	start_token_idx = word_to_index[start_token]

	unknown_lookup = [x for x in word_freq if x not in word_to_index]

	# replace uncommon words with the unknown token 
	for index,tweet in enumerate(tokenized_tweet_lst):
		tokenized_tweet_lst[index] = [twit if twit in word_to_index else unknown_token for twit in tweet]

	# map words to integers for training data
	X_train = np.asarray([[word_to_index[twit] for twit in tweet[:-1]] for tweet in tokenized_tweet_lst])
	Y_train = np.asarray([[word_to_index[twit] for twit in tweet[1:]] for tweet in tokenized_tweet_lst])

	return X_train, Y_train, index_to_word, word_to_index, vocab_size, unknown_lookup, start_token_idx, end_token_idx

# LOAD THE DATA
X_train, Y_train, index_to_word, word_to_index, vocab_size, unknown_lookup, start_token_idx, end_token_idx = preprocess()

# STEP 2: DEFINE THE NEURAL NET ARCHITECTURE
# |------------------|
# |   Output >:)     |
# |------------------|
#         ^^
#         ||
# |------------------|
# | Projection Layer |
# |------------------|
#         ^^
#         ||
# |------------------|
# |   RNN  Layer 2   |  <== Hidden State 2
# |------------------|
#         ^^
#         ||
# |------------------|
# |   RNN  Layer 1   |  <== Hidden State 1
# |------------------|
#         ^^
#         ||
# |------------------|
# |  Embedding Layer |
# |------------------|
#         ^^
#         ||
# |------------------|
# |  Input  Example  |
# |------------------|


num_epochs = 200
batch_size = 1
num_hidden_units = 100
num_layers = 2
embedding_dim = 300
num_batches = (X_train.shape[0])//batch_size
lr = 0.01

# BUILD THE TENSORFLOW GRAPH 
# |------------------|
# |  Input  Example  |
# |------------------|
X_batch = tf.placeholder(tf.int32, shape=(batch_size, None), name='input_features')
Y_batch = tf.placeholder(tf.int32, shape=(batch_size, None), name='input_label')

# cast the labels in to the appropriate shape 
Y_one_hot = tf.one_hot(Y_batch, vocab_size, on_value=1, off_value=0, name="Y_one_hot")
labels = tf.cast(Y_one_hot, tf.float32, name="training_labels")

# add an embedding layer
# |------------------|
# |  Embedding Layer |
# |------------------|
word_embeddings = tf.get_variable(name="word_embeddings", shape=(vocab_size, embedding_dim))
embedded_word_ids = tf.nn.embedding_lookup(word_embeddings, X_batch)

# define the RNN
# <== Hidden State 1
#< == Hidden State 2
LSTM_state = tf.placeholder(tf.float32, shape=(num_layers, 2, batch_size, num_hidden_units))
LSTM_state_lst = tf.unstack(LSTM_state,axis=0)
LSTM_tuple_lst = tuple([tf.nn.rnn_cell.LSTMStateTuple(LSTM_state_lst[i][0], LSTM_state_lst[i][1]) for i in range(num_layers)])		

# define the forward pass
# |------------------|
# |   RNN  Layer 2   |  <== Hidden State 2
# |------------------|
#         ^^
#         ||
# |------------------|
# |   RNN  Layer 1   |  <== Hidden State 1
# |------------------|
cells = []
for _ in range(num_layers):
	cell = tf.nn.rnn_cell.LSTMCell(num_hidden_units, state_is_tuple=True)
	cells.append(cell)

cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
outputs, H = tf.nn.dynamic_rnn(cell, embedded_word_ids, initial_state=LSTM_tuple_lst)

# last layer is fully connected to go to vocab_size
# |------------------|
# | Projection Layer |
# |------------------|
outputs = tf.reshape(outputs, shape=(-1, num_hidden_units))			
logits = tf.contrib.layers.fully_connected(outputs, vocab_size, activation_fn=None)

# |------------------|
# |   Output >:)     |
# |------------------|
logits = tf.reshape(logits, shape=(batch_size, -1, vocab_size)) # expand back out to (batch_size x max_seq_len x vocab_size)

# define loss and optimizer
loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# STEP 3: TRAIN THE MODEL
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)

	for epoch_idx in range(num_epochs):
		# shuffle the dataset
		random_indices = np.random.permutation(X_train.shape[0])
		X_train = X_train[random_indices]
		Y_train = Y_train[random_indices]

		print("epoch: {}".format(epoch_idx))
		init_LSTM_state = np.zeros((num_layers, 2, batch_size, num_hidden_units))

		for batch_idx in range(num_batches):
			if (batch_idx) %100 == 0:
				print(batch_idx)
			# run the graph
			_, minibatch_cost = sess.run([optimizer, loss], feed_dict={
																		X_batch:np.expand_dims(X_train[batch_idx], 0), 
																		Y_batch:np.expand_dims(Y_train[batch_idx], 0), 
																		LSTM_state:init_LSTM_state,
																	   })

		# STEP 4: GENERATE THE TWEETS!!!
		if epoch_idx % 5 == 0:
			# generate 20 tweets
			for _ in range(20):
				sentence = []
				counter = 0
				next_token = np.ones((batch_size,1))*start_token_idx # start token
				next_LSTM_state = np.zeros((num_layers, 2, batch_size, num_hidden_units))

				# while an end token hasn't been generated...
				while(next_token[0] != end_token_idx): 
					gen_word = index_to_word[next_token[0][0]]
					if gen_word == unknown_token:
						gen_word = unknown_lookup[np.random.randint(len(unknown_lookup))]
					sentence.append(gen_word)

					preds, next_LSTM_state = sess.run([logits, H], feed_dict={X_batch:next_token, LSTM_state:next_LSTM_state})

					# sample from probabilities
					p = tf.nn.softmax(np.squeeze(preds)).eval()
					p = p/np.sum(p)
					index = np.random.choice(vocab_size, 1, p=p)[0]

					next_token = np.ones((batch_size,1))*index
					counter += 1

				sentence = sentence[1:] # get rid of the <s> token
				print(" ".join(sentence))



