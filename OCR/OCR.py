import io
import os
import os.path as osp
import pandas as pd

import sys 
sys.path.append(os.path.dirname(os.getcwd())) 

from helpers import *
from configs import *
from google.cloud import vision
from tqdm import tqdm
from PIL import Image

print("Current directory: ", os.getcwd())
V3C1_OCR_path = osp.join(DATASET_MASTER_PATH, 'provided_ocr/V3C1_ocr')
OCR_filename = osp.join(DATASET_MASTER_PATH, 'OCR/OCR_V3C1.json')
with open(osp.join(V3C1_OCR_path, 'muchtext_keyframes.txt'), 'r') as file:
    muchtext_keyframes = file.read().splitlines()
with open(osp.join(V3C1_OCR_path, 'fewtext_keyframes.txt'), 'r') as file:
    fewtext_keyframes = file.read().splitlines()
    
text_keyframes = fewtext_keyframes + muchtext_keyframes
text_keyframes = sort_list(text_keyframes)

# Path to the Credential json (download from GG)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/home/ntnhu/projects/VideoFall-VBS2022/OCR/vbs-ocr.json" 

def convert_response_to_list(res):
    texts = res.text_annotations
    result = []
    for text in texts:
        di = dict()
        di['text'] = text.description
        di['bb'] = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
        result.append(di)
    return result

# Declare API
client = vision.ImageAnnotatorClient()

OCR_dict = {}
for img_id in tqdm(text_keyframes[:10]):
    path = osp.join(KEYFRAME_PATH, img_id)

    with io.open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    # Run API (its free 1000 times/month --> later 1.5 / 1000 calls)
    try:
        response = client.text_detection(image=image)
        texts = [text.description for text in response.text_annotations]
        ocr = ' '.join(texts)
        ocr = ocr.replace('\n', ' ')
        ocr = re.sub(' +', ' ', ocr)
        OCR_dict[img_id] = ocr
    except:
        print(f"There is a problem with {img_id}")
        
with open(OCR_filename, 'w') as fp:
    json.dump(OCR_dict, fp, indent=4)