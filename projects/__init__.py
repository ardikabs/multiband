
from flask import Flask, render_template, request, url_for,redirect, jsonify


app = Flask(__name__)


@app.route('/')
def index():
    return jsonify({
        "status" : "OK",
        "message" : "Multiband Image Clustering by Ardika Bagus Saputro 2017"
    })
    
@app.route('/multiband')
def multiband_index():
    return render_template('multiband/index.html',title="Multiband Image Clustering")