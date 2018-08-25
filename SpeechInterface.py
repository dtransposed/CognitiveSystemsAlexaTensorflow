# some imports 
import speech_recognition as sr
import tensorflow as tf
import pickle
import numpy as np
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def record_voice(computational_graph,embed):
    """
	does a voice recording and converts it to speech via google api
	if current_command obtained from audio signal is not equal to
	none we update the current command in command_file.txt
	"""

    r = sr.Recognizer()

    mic = sr.Microphone()

    # get audio from mic
    with mic as source:
        r.adjust_for_ambient_noise(source)  # helps us to deal with the background noise
        print('Speak please...')
        audio = r.listen(source, timeout=10)  # records the voice from the microphone
        print('Stop please...')

    # convert to text via google api
    try:
        text = r.recognize_google(audio, language='en-GB')  # uses Google Speech API to convert voice into text (str)
    except:
        text = 'none'

    current_command = text.lower()

    if current_command != 'none':

        dictionary = pickle.load(open("lookup_table.p", "rb"))
        lookup_table = np.array(list(dictionary.values()))
        command_list = list(dictionary.keys())


        with tf.Session(graph = computational_graph) as session:

            messages = [current_command]
            output = embed(messages)

            start = time.time()

            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            message_embeddings = session.run(output)
            index = np.argmax(np.inner(lookup_table, message_embeddings))
            current_command = command_list[index]
            print(current_command)

            done = time.time()
            elapsed = done - start
            print('Time elapsed {}'.format(elapsed))

        # update command file
        with open("command_file.txt", "w") as command_file:
            if current_command == 'stop':
                command_file.write('none')
            else:
                command_file.write(current_command)