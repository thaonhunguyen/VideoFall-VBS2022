import io
import os
import os.path as osp
import pandas as pd
import argparse
import sys 
sys.path.append(os.path.dirname(os.getcwd())) 

from helpers import *
from configs import *
from google.cloud import vision
from google.cloud.vision_v1 import AnnotateImageResponse
from tqdm import tqdm
from PIL import Image

print("Current directory: ", os.getcwd())

# Path to the Credential json (download from GG)
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/home/ntnhu/projects/VideoFall-VBS2022/OCR/vbs-ocr.json" 
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/home/nmduy/OCR_LSC21/lsc2021-freetrial.json" 

parser = argparse.ArgumentParser(description="VBS22 Interactive Video Search Engine")
parser.add_argument('--dataset_name', '-d', type=str, help='Name of the current dataset')
parser.add_argument('--part', '-p', type=int, default=1)
parser.add_argument('--credential_path', '-cp', type=str, help='Path to the Credential json (download from gg)')


# Define some functions for later use
def run_ggapi_ocr(img_path, client):
    # Read image
    with io.open(img_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    # Run API (its free 1000 times/month --> later 1.5 / 1000 calls)
    response = client.text_detection(image=image)
    return response

def convert_response_to_list(res):
    texts = res.text_annotations
    result = []
    for text in texts:
        di = dict()
        di['text'] = text.description
        di['bb'] = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
        result.append(di)
    return result

def process_text(text_list):
    text = ' '.join(text_list)
    text = text.replace('\n', ' ')
    text = re.sub(' +', ' ', text)
    return text

def main(args):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = args.credential_path
    # Determine those keyframe containing text
    if args.dataset_name == 'V3C1':
        V3C1_OCR_path = osp.join(DATASET_MASTER_PATH, 'provided_ocr/V3C1_ocr')
        OCR_filename = osp.join(DATASET_MASTER_PATH, f'OCR/OCR_V3C1_{args.part}.json')
        OCR_text_filename = osp.join(DATASET_MASTER_PATH, f'OCR/OCR_V3C1_text_{args.part}.json')
        with open(osp.join(V3C1_OCR_path, 'muchtext_keyframes.txt'), 'r') as file:
            muchtext_keyframes = file.read().splitlines()
        with open(osp.join(V3C1_OCR_path, 'fewtext_keyframes.txt'), 'r') as file:
            fewtext_keyframes = file.read().splitlines()
        text_keyframes = fewtext_keyframes + muchtext_keyframes
        text_keyframes = sort_list(text_keyframes)
    
    # Declare API
    client = vision.ImageAnnotatorClient()
    OCR_text_dict = {}
    OCR_dict = {}
    index = (args.part - 1)*30000 + 50000

    for img_id in tqdm(text_keyframes[index:index+30000]):
        response = run_ggapi_ocr(osp.join(KEYFRAME_PATH, img_id), client)
        ocr_result = convert_response_to_list(response)
        ocr_original_json = AnnotateImageResponse.to_json(response)
        ocr_original_json = json.loads(ocr_original_json)
        texts = [text.description for text in response.text_annotations]
        OCR_dict[img_id] = ocr_original_json
        OCR_text_dict[img_id] = process_text(texts)
#         except:
#             print(f"There is a problem with {img_id}")

    with open(OCR_filename, 'w') as fp:
        json.dump(OCR_dict, fp, indent=4)
    with open(OCR_text_filename, 'w') as fp:
        json.dump(OCR_text_dict, fp, indent=4)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)