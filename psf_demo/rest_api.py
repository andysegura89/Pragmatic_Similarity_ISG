from datetime import datetime
import json
import time
import threading
import os

import librosa
import flask
from flask import Flask, Response, jsonify
from flask_restful import Api
from flask_cors import CORS
from playsound import playsound

import similarity_finder as sf
import audio_recorder

recording_semaphore = threading.Semaphore()
recording_semaphore.acquire()
app = Flask("PSFAPI")
CORS(app)
api = Api(app)
recording = False
similarity_finder = sf.SimilarityFinder(feature_selection=True)

clips = []
finished_clips = {}
recently_updated_clips = []

num_clips_processing = 0
current_dataset = 'DRAL'

saved_clips = [
{'id': 51, 'path': '02_21_16_23_31.wav', 'duration': librosa.get_duration(path='clips/02_21_16_23_31.wav'), 'dataset': 'DRAL', 'best_path': '/Users/andy/Desktop/UTEP/Fall23/Research/DRAL-All-Short/EN_025_24.wav', 'best_cos': 0.74, 'best_asd_label': None, 'second_path': '/Users/andy/Desktop/UTEP/Fall23/Research/DRAL-All-Short/EN_001_3.wav', 'second_cos': 0.7, 'second_asd_label': None, 'hundred_path': '/Users/andy/Desktop/UTEP/Fall23/Research/DRAL-All-Short/EN_058_9.wav', 'hundred_cos': 0.5, 'hundred_asd_label': None, 'five_hundred_path': '/Users/andy/Desktop/UTEP/Fall23/Research/DRAL-All-Short/EN_021_25.wav', 'five_hundred_cos': 0.48, 'five_hundred_asd_label': None},
{'id': 42, 'path': '02_21_16_22_42.wav', 'duration': librosa.get_duration(path='clips/02_21_16_22_42.wav'), 'dataset': 'DRAL', 'best_path': '/Users/andy/Desktop/UTEP/Fall23/Research/DRAL-All-Short/EN_020_12.wav', 'best_cos': 0.71, 'best_asd_label': None, 'second_path': '/Users/andy/Desktop/UTEP/Fall23/Research/DRAL-All-Short/EN_026_12.wav', 'second_cos': 0.7, 'second_asd_label': None, 'hundred_path': '/Users/andy/Desktop/UTEP/Fall23/Research/DRAL-All-Short/EN_019_41.wav', 'hundred_cos': 0.39, 'hundred_asd_label': None, 'five_hundred_path': '/Users/andy/Desktop/UTEP/Fall23/Research/DRAL-All-Short/EN_039_19.wav', 'five_hundred_cos': 0.35, 'five_hundred_asd_label': None},
{'id': 40, 'path': '02_21_16_22_36.wav', 'duration': librosa.get_duration(path='clips/02_21_16_22_36.wav'), 'dataset': 'DRAL', 'best_path': '/Users/andy/Desktop/UTEP/Fall23/Research/DRAL-All-Short/EN_059_18.wav', 'best_cos': 0.9, 'best_asd_label': None, 'second_path': '/Users/andy/Desktop/UTEP/Fall23/Research/DRAL-All-Short/EN_009_55.wav', 'second_cos': 0.9, 'second_asd_label': None, 'hundred_path': '/Users/andy/Desktop/UTEP/Fall23/Research/DRAL-All-Short/EN_003_37.wav', 'hundred_cos': 0.7, 'hundred_asd_label': None, 'five_hundred_path': '/Users/andy/Desktop/UTEP/Fall23/Research/DRAL-All-Short/EN_029_24.wav', 'five_hundred_cos': 0.63, 'five_hundred_asd_label': None},
{'id': 36, 'path': '02_21_16_21_59.wav', 'duration': librosa.get_duration(path='clips/02_21_16_21_59.wav'), 'dataset': 'DRAL', 'best_path': '/Users/andy/Desktop/UTEP/Fall23/Research/DRAL-All-Short/EN_103_3.wav', 'best_cos': 0.92, 'best_asd_label': None, 'second_path': '/Users/andy/Desktop/UTEP/Fall23/Research/DRAL-All-Short/EN_086_21.wav', 'second_cos': 0.91, 'second_asd_label': None, 'hundred_path': '/Users/andy/Desktop/UTEP/Fall23/Research/DRAL-All-Short/EN_005_37.wav', 'hundred_cos': 0.72, 'hundred_asd_label': None, 'five_hundred_path': '/Users/andy/Desktop/UTEP/Fall23/Research/DRAL-All-Short/EN_089_16.wav', 'five_hundred_cos': 0.65, 'five_hundred_asd_label': None},
{'id': 34, 'path': '02_21_16_21_38.wav', 'duration': librosa.get_duration(path='clips/02_21_16_21_38.wav'), 'dataset': 'DRAL', 'best_path': '/Users/andy/Desktop/UTEP/Fall23/Research/DRAL-All-Short/EN_073_23.wav', 'best_cos': 0.87, 'best_asd_label': None, 'second_path': '/Users/andy/Desktop/UTEP/Fall23/Research/DRAL-All-Short/EN_093_4.wav', 'second_cos': 0.86, 'second_asd_label': None, 'hundred_path': '/Users/andy/Desktop/UTEP/Fall23/Research/DRAL-All-Short/EN_044_17.wav', 'hundred_cos': 0.67, 'hundred_asd_label': None, 'five_hundred_path': '/Users/andy/Desktop/UTEP/Fall23/Research/DRAL-All-Short/EN_019_51.wav', 'five_hundred_cos': 0.61, 'five_hundred_asd_label': None},
{'id': 33, 'path': '02_21_16_21_28.wav', 'duration': librosa.get_duration(path='clips/02_21_16_21_28.wav'), 'dataset': 'DRAL', 'best_path': '/Users/andy/Desktop/UTEP/Fall23/Research/DRAL-All-Short/EN_075_4.wav', 'best_cos': 0.86, 'best_asd_label': None, 'second_path': '/Users/andy/Desktop/UTEP/Fall23/Research/DRAL-All-Short/EN_078_12.wav', 'second_cos': 0.85, 'second_asd_label': None, 'hundred_path': '/Users/andy/Desktop/UTEP/Fall23/Research/DRAL-All-Short/EN_104_12.wav', 'hundred_cos': 0.68, 'hundred_asd_label': None, 'five_hundred_path': '/Users/andy/Desktop/UTEP/Fall23/Research/DRAL-All-Short/EN_006_8.wav', 'five_hundred_cos': 0.62, 'five_hundred_asd_label': None},
{'id': 32, 'path': '02_21_16_21_18.wav', 'duration': librosa.get_duration(path='clips/02_21_16_21_18.wav'), 'dataset': 'DRAL', 'best_path': '/Users/andy/Desktop/UTEP/Fall23/Research/DRAL-All-Short/EN_075_4.wav', 'best_cos': 0.89, 'best_asd_label': None, 'second_path': '/Users/andy/Desktop/UTEP/Fall23/Research/DRAL-All-Short/EN_084_1.wav', 'second_cos': 0.89, 'second_asd_label': None, 'hundred_path': '/Users/andy/Desktop/UTEP/Fall23/Research/DRAL-All-Short/EN_021_20.wav', 'hundred_cos': 0.72, 'hundred_asd_label': None, 'five_hundred_path': '/Users/andy/Desktop/UTEP/Fall23/Research/DRAL-All-Short/EN_008_53.wav', 'five_hundred_cos': 0.66, 'five_hundred_asd_label': None},
{'id': 31, 'path': '02_21_16_21_07.wav', 'duration': librosa.get_duration(path='clips/02_21_16_21_07.wav'), 'dataset': 'DRAL', 'best_path': '/Users/andy/Desktop/UTEP/Fall23/Research/DRAL-All-Short/EN_080_8.wav', 'best_cos': 0.86, 'best_asd_label': None, 'second_path': '/Users/andy/Desktop/UTEP/Fall23/Research/DRAL-All-Short/EN_039_28.wav', 'second_cos': 0.86, 'second_asd_label': None, 'hundred_path': '/Users/andy/Desktop/UTEP/Fall23/Research/DRAL-All-Short/EN_037_15.wav', 'hundred_cos': 0.68, 'hundred_asd_label': None, 'five_hundred_path': '/Users/andy/Desktop/UTEP/Fall23/Research/DRAL-All-Short/EN_070_13.wav', 'five_hundred_cos': 0.62, 'five_hundred_asd_label': None},
{'id': 54, 'path': '02_21_16_28_17.wav', 'duration': librosa.get_duration(path='clips/02_21_16_28_17.wav'), 'dataset': 'DRAL', 'best_path': '/Users/andy/Desktop/UTEP/Fall23/Research/DRAL-All-Short/EN_038_25.wav', 'best_cos': 0.89, 'best_asd_label': None, 'second_path': '/Users/andy/Desktop/UTEP/Fall23/Research/DRAL-All-Short/EN_074_9.wav', 'second_cos': 0.88, 'second_asd_label': None, 'hundred_path': '/Users/andy/Desktop/UTEP/Fall23/Research/DRAL-All-Short/EN_004_30.wav', 'hundred_cos': 0.57, 'hundred_asd_label': None, 'five_hundred_path': '/Users/andy/Desktop/UTEP/Fall23/Research/DRAL-All-Short/EN_063_25.wav', 'five_hundred_cos': 0.54, 'five_hundred_asd_label': None},
{'id': 58, 'path': '02_21_16_28_50.wav', 'duration': librosa.get_duration(path='clips/02_21_16_28_50.wav'), 'dataset': 'DRAL', 'best_path': '/Users/andy/Desktop/UTEP/Fall23/Research/DRAL-All-Short/EN_098_23.wav', 'best_cos': 0.84, 'best_asd_label': None, 'second_path': '/Users/andy/Desktop/UTEP/Fall23/Research/DRAL-All-Short/EN_013_32.wav', 'second_cos': 0.82, 'second_asd_label': None, 'hundred_path': '/Users/andy/Desktop/UTEP/Fall23/Research/DRAL-All-Short/EN_085_14.wav', 'hundred_cos': 0.59, 'hundred_asd_label': None, 'five_hundred_path': '/Users/andy/Desktop/UTEP/Fall23/Research/DRAL-All-Short/EN_023_9.wav', 'five_hundred_cos': 0.56, 'five_hundred_asd_label': None},
]

for clip in saved_clips:
    finished_clips['clips/' + clip['path']] = [
        [None, clip['best_path']],
        [None, clip['second_path']],
        [None, clip['hundred_path']],
        [None, clip['five_hundred_path']]
    ]


def asd_finder(path):
    global current_dataset
    if current_dataset != 'ASDNT':
        return None
    return 'ASD' if 'ASD-Mono' in path else 'NT'


def find_similar_clips(id, path):
    global finished_clips, recently_updated_clips, current_dataset, num_clips_processing
    first_place, second_place, hundred_place, five_hundred = \
        similarity_finder.find_similar(path, current_dataset)
    finished_clips[path] = [first_place, second_place, hundred_place, five_hundred]
    clip = {'id': id,
            'path': path[6:],
            'duration': None,
            'dataset': current_dataset,
            'best_path': first_place[1],
            'best_cos': round(first_place[0].item(), 2),
            'best_asd_label': asd_finder(first_place[1]),
            'second_path': second_place[1],
            'second_cos': round(second_place[0].item(), 2),
            'second_asd_label': asd_finder(second_place[1]),
            'hundred_path': hundred_place[1],
            'hundred_cos': round(hundred_place[0].item(), 2),
            'hundred_asd_label': asd_finder(hundred_place[1]),
            'five_hundred_path': five_hundred[1],
            'five_hundred_cos': round(five_hundred[0].item(), 2),
            'five_hundred_asd_label': asd_finder(five_hundred[1])
            }
    print(f"\n{clip}\n")
    recently_updated_clips.append(clip)
    num_clips_processing -= 1

@app.route('/load_clips/', methods=['GET'])
def load_saved_clips():
    global saved_clips
    return jsonify(saved_clips)


@app.route('/stream/')
def stream():
    global recording
    def get_clips():
        global recording, recording_semaphore, recently_updated_clips, num_clips_processing
        while True:
            recording_semaphore.acquire()
            now = datetime.now()
            file_path = f'clips/{now.strftime("%m_%d_%H_%M_%S")}.wav'
            audio_recorder.run(file_path)
            clip_id = len(clips) + 1
            duration = librosa.get_duration(path=file_path)
            clip = {'id': clip_id,
                    'path': file_path[6:],
                    'duration': duration,
                    'dataset': current_dataset,
                    'best_path': None,
                    'best_cos': None,
                    'second_path': None,
                    'second_cos': None,
                    'hundred_path': None,
                    'hundred_cos': None,
                    'five_hundred_path': None,
                    'five_hundred_cos': None
                    }
            clips.append(clip)
            work_thread = threading.Thread(target=find_similar_clips, args=[clip_id, file_path])
            work_thread.start()
            recording_semaphore.release()
            if recording:
                num_clips_processing += 1
                yield f"data: {json.dumps(clip)}\n\n"
            while not recording and num_clips_processing != 0:
                time.sleep(0.1)
            while len(recently_updated_clips) > 0:
                clip = recently_updated_clips.pop(0)
                yield f"data: {json.dumps(clip)}\n\n"
    r = Response(get_clips(), mimetype='text/event-stream')
    r.headers.add('Access-Control-Allow-Origin', '*')
    return r

@app.route('/toggle_recording/<dataset>', methods=['PUT'])
def toggle_recording(dataset):
    global recording, recording_semaphore, current_dataset
    if not recording:
        current_dataset = dataset
        recording_semaphore.release()
        recording = True
    else:
        recording_semaphore.acquire()
        recording = False
    response = flask.jsonify({"recording":recording})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/play_clip/<path>', methods=['PUT'])
def play_clip(path):
    global finished_clips
    print(f"playing_path={path}")
    flag = path[:3]
    if flag == 'fp-':
        path = finished_clips['clips/' + path[3:]][0][1]
    elif flag == 'sp-':
        path = finished_clips['clips/' + path[3:]][1][1]
    elif flag == '1p-':
        path = finished_clips['clips/' + path[3:]][2][1]
    elif flag == '5p-':
        path = finished_clips['clips/' + path[3:]][3][1]
    else:
        path = 'clips/' + path
    if os.path.exists(path):
        playsound(path)
    else:
        print(f"file path not found when trying to play: {path}")
    response = flask.jsonify({"played": True})
    return response


if __name__ == '__main__':
    app.run(port=5050)
