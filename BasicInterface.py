# some imports
import tkinter as tk
import SpeechInterface as SI
import tensorflow as tf
import tensorflow_hub as hub
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

computational_graph = tf.Graph()

with computational_graph.as_default():
    embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")


class GUI:
    """
	very simple user interface class
	"""

    def __init__(self, master, computational_graph, embed):
        """
		creates an interface consisting of
		two buttons - quit and record

		Args:
			master: of type Tk. Parent UI element
		"""

        self.master = master
        master.title("Simple User-Interface")
        self.graph = computational_graph
        self.embed = embed

        self.close_button = tk.Button(master, text="QUIT", command=self.master.quit)
        self.close_button.pack()

        self.record_button = tk.Button(master, text="RECORD", command=self.run_recording)
        self.record_button.pack()

    def run_recording(self):
        """
		calls the recording routine
		"""
        SI.record_voice(self.graph, self.embed)


# initialize command to none
with open("command_file.txt", "w") as command_file:
    command_file.write('none')

# initialize UI
root = tk.Tk()
frame = GUI(root, computational_graph, embed)

# run
root.mainloop()