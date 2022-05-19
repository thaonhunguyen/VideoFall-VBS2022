import sys
from dataset_helpers import *
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np 
import faiss
import time
import joblib
import csv
import subprocess
from configs import *
import pickle

generate_features = True

print("Dataset name: ", DATASET_NAME, file=sys.stdout)
clip = CLIPSearchEngine(src_path=DATASET_MASTER_PATH, feature_path=FEATURE_PATH, generate_features=generate_features)
clip.dataset.get_file_name()
print('Loading Features', file=sys.stdout)
clip.load_features()

with open("feature_dict.pkl", "rb") as a_file:
    print('Loading Feature Dict', file=sys.stdout)
    clip.feature_dict = pickle.load(a_file)
    a_file.close()
    

from flask import Flask, request

app = Flask(__name__)


def format_result(result_entity):
    """
    Format result to get just video id and shot id
    """
    return {
        # "path": result_entity['path'],
        "video": result_entity['video'],
        "shot": result_entity['shot'],
    }


@app.route('/api/search', methods=['POST'])
def search():
    body = request.get_json()
    query = body['query'] if 'query' in body else ''
    ocr = body['ocr'] if 'ocr' in body else ''
    colors = body['colors'] if 'colors' in body else []
    best_images = clip.search_query(query, num_matches=1000, ss_type='other')
    
    return {
        "result": list(map(format_result, best_images)),
    }


@app.route('/api/find_similar_keyframes/<video_id>/<keyframe_id>', methods=['GET'])
def similar_keyframes(video_id, keyframe_id):
    # query = f'/mnt/deakin/VBS2022/keyframes/{video_id}/shot{video_id}_{keyframe_id}_RKF.png'
    # img_query = convert_to_concepts(query, dataset_name=DATASET_NAME)['filename']
    img_query = f'shot{video_id}_{keyframe_id}_RKF.png'
    feature = clip.feature_dict[img_query]
    feature_vec = np.expand_dims(feature, axis=0)
    feature_vector = feature_vec.astype('float32')
    similarities = (feature_vector @ clip.features.T).squeeze(0)
    indices = similarities.argsort()[-50:][::-1]
    best_matched_image_names = [clip.dataset.image_names[item] for item in indices]

    result = [convert_to_concepts(item, dataset_name=clip.dataset_name) for item in best_matched_image_names]
    return {
        "result": list(map(format_result, result[:1000])),
    }
    
    
if __name__ == '__main__':
    print('Running server', file=sys.stdout)
    app.run(host='0.0.0.0', port=6007, debug=True)