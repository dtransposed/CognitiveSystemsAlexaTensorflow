import tensorflow_hub as hub
import tensorflow as tf
import collections
import pickle

print("Creating a lookup table...")
'''
First part is downloading the model. It takes some time, so in order to track the installation
one may set a logger.
'''
#optional if we want to track progress
tf.logging.set_verbosity(10)

#This model is downloaded once and then cached in the local system temporary directory
sent2vec = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")

#Those are test commands to be executed by the robot

"""
Here we can define the set of commands, which flow into nrp as strings .e.g
commands = ['Turn left', 'Turn right', 'Go forward', 'Go backward', 'Stop!', 'Go to the red object','Go to the blue object']
"""
commands = ['go', 'back', 'stop']

with tf.Session() as session:
  session.run([tf.global_variables_initializer(), tf.tables_initializer()])
  message_embeddings = session.run(sent2vec(commands))

dictionary = collections.OrderedDict(zip(commands, message_embeddings))
pickle.dump(dictionary, open( "lookup_table.p", "wb" ) )
print("Lookup table saved as lookup_table.p")