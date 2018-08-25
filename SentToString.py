import tensorflow_hub as hub
import tensorflow as tf
import time
import pickle
import numpy as np

dictionary = pickle.load(open( "save.p", "rb" ) )
lookup_table = np.array(list(dictionary.values()))
command_list = list(dictionary.keys())

with tf.Graph().as_default():
  embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")
  messages = tf.placeholder(dtype=tf.string, shape=[None])
  output = embed(messages)

  with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    while True:
      sentence = input('Enter your input:')
      t1 = time.time()
      message_embeddings = session.run(output, feed_dict={messages: [sentence]})
      index = np.argmax(np.inner(lookup_table, message_embeddings))
      print(command_list[index])
      print('Time of inference: ',time.time() - t1)


