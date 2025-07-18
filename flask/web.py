from flask import Flask, render_template, jsonify
import json
import os

app = Flask(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

@app.route('/')
def index():
    return render_template('ParkingMap2.html')

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
    # 仮の配列を直接返す
    # -1: 灰色, 0: 赤色, 1: 黄色, 2: 緑色
    # dummy_data = [2, -1, -1, -1, -1, -1, 0, -1, 1]  # 長さ9の配列
    dummy_data = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # 長さ9の配列
    return jsonify(dummy_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)