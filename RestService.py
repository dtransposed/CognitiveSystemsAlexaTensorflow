from flask import Flask

app = Flask(__name__)

@app.route("/")
def get_command():
	with open("command_file.txt", "r") as command_file:
		current_command = command_file.read()
	return current_command


if __name__ == '__main__':
    app.run(host='0.0.0.0')