from flask import Flask, render_template, jsonify
import json
import os

app = Flask(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

@app.route('/')
def index():
    return render_template('ParkingMap.html')

@app.route('/archive')
def archive():
    return render_template('ParkingMapArchive.html')

@app.route('/api/calculateparking')
def api_calculate_parking():
    path = os.path.join(DATA_DIR, 'calculateparking.json')
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return jsonify({"error": "Failed to load data"}), 500
    return jsonify(data)

@app.route('/api/parking_stats')
def calculate_parking():
    path = os.path.join(DATA_DIR, 'parking_stats.json')
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
