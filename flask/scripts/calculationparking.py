import csv
import json
import glob
import os
from collections import defaultdict

# 曜日と時間帯の対応表
WEEKDAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
TIME_SLOTS = ['08:40', '10:20', '12:20', '12:50', '14:20', '16:00']

# データ構造：{day: {time: [ [list of lists] ] }}
aggregated = defaultdict(lambda: defaultdict(list))

# ディレクトリのパス
BASE_DIR = os.path.dirname(__file__)
ARCHIVE_DIR = os.path.join(BASE_DIR, '../data/ParkingArchive')
OUTPUT_FILE = os.path.join(BASE_DIR, '../data/calculateparking.json')

# CSVファイル読み込み
for filepath in glob.glob(os.path.join(ARCHIVE_DIR, "*.csv")):
    with open(filepath, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 11:
                continue  # 不正行スキップ

            weekday_idx = int(row[0])
            time_idx = int(row[1])
            slots = list(map(int, row[2:11]))

            day = WEEKDAYS[weekday_idx]
            time = TIME_SLOTS[time_idx]

            aggregated[day][time].append(slots)

# 平均を計算してJSON形式で出力
result = {}
for day, times in aggregated.items():
    result[day] = {}
    for time, values in times.items():
        n = len(values)
        avg = [round(sum(slot[i] for slot in values) / n, 2) for i in range(9)]
        result[day][time] = avg

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print("✅ parking_stats.json を出力しました。")