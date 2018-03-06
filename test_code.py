import numpy as np
from string import punctuation
import nltk
from nltk.probability import FreqDist
from wordcloud import WordCloud
import matplotlib as mpl
from collections import Counter
import tensorflow as tf
mpl.use("TkAgg")
import matplotlib.pyplot as plt
with open('reviews.txt') as f:
    reviews = f.readlines()
    print len(reviews)




with open('labels.txt') as f:
    labels = f.read()
    print len(labels)


review_list = []

for lines in reviews:
    review_list.append(lines)


all_text = ' '.join([c for c in review_list if c not in punctuation])


review_list = all_text.split('\n')
# review_list.pop(-1)

print "Length of review is {}".format(len(review_list))


# all_text = ' '.join(reviews)

words = all_text.split()
# fdist = FreqDist(words)
# top_ten = fdist.most_common(10)

# wordcloud = WordCloud(max_words=10).generate_from_frequencies(WordCloud().process_text(all_text))
# # print wordcloud
# plt.figure()
# plt.imshow(wordcloud, interpolation="bilinear")
# plt.axis("off")
# plt.show()



######### Converting words to integers
# if words[0]=="bromwell":
# 	print "YASS"
counts = Counter(words)

vocab = sorted(counts,key=counts.get,reverse=True)
vocab_int = {word : index for index,word in enumerate(vocab,1)}
int_vocab = {index : word for index,word in enumerate(vocab,1)}
# print vocab_int['bromwell']

review_ints = []

for l in review_list:
	review_ints.append([vocab_int[word] for word in l.split()])


########Converting labels into integers


labels_int = np.array([1 if l=="positive" else 0 for l in labels.split()])
print labels_int[:2]
# print review_ints[0]

print labels_int[:100]
######## Checking for zero length reviews

review_lens = Counter([len(x) for x in review_ints])

print "Zero length reviews are {}".format(review_lens[0])
print "Maximum review length is {}".format(max(review_lens))

######### Filtering out rreviews with zero length and truncating those with more than 200 length

review_ints = [r[:200] for r in review_ints if len(r)>0]

review_lens = Counter([len(x) for x in review_ints])
print "Zero length reviews are {}".format(review_lens[0])
print "Maximum review length is {}".format(max(review_lens))


#### Populating the feature list
seq_len = 200
features = np.zeros([len(review_ints),seq_len],dtype=int)

for index,row in enumerate(review_ints):
	features[index, -len(row):] = np.array(row)[:seq_len]


# print(interesting)
# print(len(features))
# print(type(features))
# print(features[41])
# print(len(features[41]))
# print(review_ints[41])
# print(len(review_ints[41]))

# print features[:10,:100]


print "Length of review int 0 is {}".format(len(review_ints[0]))
split_frac_val = 0.8
split_index_val = int(len(features) * split_frac_val)
# print len(features)
# print len(review_ints)

train_x, val_x = features[:split_index_val], features[split_index_val:]
train_y , val_y = labels_int[:split_index_val], labels_int[split_index_val:]

split_frac = 0.5
split_index = int(len(val_x) * split_frac)

val_x, test_x = val_x[:split_index], val_x[split_index:]
val_y, test_y = val_y[:split_index], val_y[split_index:]

# print len(val_x)
# print len(test_x)
# print len(val_y)
# print len(test_y)

# num = np.random.randint(len(test_x))
# print num
# print len(train_x[num])
# print len(test_y[num])
print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape), 
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))
print("label set: \t\t{}".format(train_y.shape), 
      "\nValidation label set: \t{}".format(val_y.shape),
      "\nTest label set: \t\t{}".format(test_y.shape))

####### Building the model graph

lstm_size = 256
lstm_layers = 2
batch_size = 1000
learning_rate = 0.01

n_words = len(vocab_int) + 1

tf.reset_default_graph()

with tf.name_scope('inputs'):
	input_ = tf.placeholder(tf.int32, [None,None], name="inputs")
	labels_ = tf.placeholder(tf.int32, [None,None], name="labels")
	keep_prob = tf.placeholder(tf.float32, name = "keep_prob")


embedding_size = 300

with tf.name_scope('Embeddings'):
	embedding = tf.Variable(tf.random_uniform((n_words,embedding_size),-1,1))
	embed = tf.nn.embedding_lookup(embedding,input_)


def lstm_cell():
	# Basic LSTM cell
	lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size,reuse = tf.get_variable_scope().reuse)

	# Adding the dropout
	return tf.contrib.rnn.DropoutWrapper(lstm,output_keep_prob = keep_prob)


with tf.name_scope("RNN_layers"):
	cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(lstm_layers)])

	initial_state = cell.zero_state(batch_size,tf.float32)


with tf.name_scope("RNN_Forward"):
	output, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state = initial_state)

with tf.name_scope("Predictions"):
	predictions = tf.contrib.layers.fully_connected(output[:,-1], 1, activation_fn = tf.sigmoid)
	tf.summary.histogram('predictions', predictions)


with tf.name_scope("Cost"):
	cost = tf.losses.mean_squared_error(labels_, predictions)
	tf.summary.scalar('cost', cost)

with tf.name_scope('train'):
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


merged = tf.summary.merge_all()


########## Validation Accuracy

with tf.name_scope('Validation'):
	correct_pred = tf.equal(tf.cast(tf.round(predictions),tf.int32), labels_)
	accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))


########### Batching
def get_batches(x,y, batch_size = 1000):
	n_batches = len(x)//batch_size

	x,y = x[:n_batches*batch_size], y[:n_batches*batch_size]

	for ii in range(0,len(x),batch_size):
		yield x[ii:ii+batch_size], y[ii:ii+batch_size]



############# Training


epochs = 10

saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	train_writer = tf.summary.FileWriter("./logs/tb/train",sess.graph)
	test_writer = tf.summary.FileWriter("./logs/tb/test",sess.graph)

	iteration = 1

	for e in range(epochs):

		state = sess.run(initial_state)

		for index, (x,y) in enumerate(get_batches(train_x, train_y, batch_size),1):
			feed = {input_: x,
					labels_: y[:, None],
					keep_prob: 0.5,
					initial_state: state}

			summary, loss, state, _ = sess.run([merged, cost, final_state, optimizer], feed_dict = feed)

			train_writer.add_summary(summary, iteration)

			if iteration%5==0:
				print("Epoch : {}/{}".format(e,epochs),
					  "Iteration: {}".format(iteration),
					   "Train loss: {:.3f}".format(loss))

			if iteration%25==0:
				val_acc = []
				val_state = sess.run(cell.zero_state(batch_size,tf.float32))

				for x,y in get_batches(val_x, val_y, batch_size):
					feed = {input_: x,
							labels_: y[:, None],
							keep_prob:1,
							initial_state:val_state }


					summary,batch_acc,val_state = sess.run([merged, cost, final_state], feed_dict = feed)

					val_acc.append(batch_acc)

				print("Val acc: {:.3f}".format(np.mean(val_acc)))

			iteration+=1
			test_writer.add_summary(summary,iteration)
			saver.save(sess, "./checkpoints/sentiment_arnav.ckpt")

	saver.save(sess, "./checkpoints/sentiment_arnav.ckpt")



# test_acc = []
# with tf.Session() as sess:
#     saver.restore(sess, "checkpoints/sentiment_arnav.ckpt")
#     test_state = sess.run(cell.zero_state(batch_size, tf.float32))
#     for ii, (x, y) in enumerate(get_batches(test_x, test_y, batch_size), 1):
#         feed = {input_: x,
#                 labels_: y[:, None],
#                 keep_prob: 1,
#                 initial_state: test_state}
#         batch_acc, test_state = sess.run([accuracy, final_state], feed_dict=feed)
#         test_acc.append(batch_acc)
#     print("Test accuracy: {:.3f}".format(np.mean(test_acc)))

print "printing text 4"
print test_x[4]

testing = "The book was very good. The writer wrote a brilliant and amazing script. It was the best"
# testing = ' '.join([testing if testing not in punctuation])
testing = testing.translate(None,punctuation)

testing_int = [vocab_int[word.lower()] for word in testing.split()]
int_testing = [int_vocab[index] for index in testing_int]






print testing_int
# print int_testing
feature = np.zeros([1,200],dtype=int)

feature[0,-len(testing_int):] = np.array(testing_int)[:seq_len] 
# print feature[0].shape

# with tf.Session() as sess:
# 	saver.restore(sess, "checkpoints/sentiment_arnav.ckpt")
# 	test_state = sess.run(cell.zero_state(batch_size,tf.float32))
	# print sess.run([final_state],feed_dict = {input_:feature[0], keep_prob:1,initial_state:test_state})

# for ii, (x, y) in enumerate(get_batches(test_x, test_y, batch_size), 1):
# 	print x.shape
# 	print x[-5:]
# 	break

# test_acc = []
# with tf.Session() as sess:
#     saver.restore(sess, "checkpoints/sentiment_arnav.ckpt")
#     test_state = sess.run(cell.zero_state(batch_size, tf.float32))
#     for ii, (x, y) in enumerate(get_batches(test_x, test_y, batch_size), 1):
#     	x[-1] = feature
#     	a = [int_vocab[c] for c in x[-10] if c not in [0]]
#     	print a
#         feed = {input_: x,
#                 labels_: y[:, None],
#                 keep_prob: 1,
#                 initial_state: test_state}
#         # batch_acc, test_state = sess.run([accuracy, final_state], feed_dict=feed)
#         print sess.run([predictions],feed_dict = {input_:x, keep_prob:1,initial_state:test_state})
#         # test_acc.append(batch_acc)
#         break
    # print("Test accuracy: {:.3f}".format(np.mean(test_acc)))




