from flask import Flask
import os

# Define upload folder
UPLOAD_FOLDER = '/pub'

# Create a Flask app
app = Flask(__name__)

# Generate some random keys for separate sessions
app.config['SECRET_KEY'] = os.urandom(16)

# Setup upload folder
app.config['UPLOAD_FOLDER'] = os.getcwd() + UPLOAD_FOLDER

# Setup maximum uploadeble file size: 32 megabytes
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

# Session type
app.config['SESSION_TYPE'] = 'filesystem'
