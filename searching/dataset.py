from configs import *
from helpers import *
from PIL import Image
from typing import List
from collections import defaultdict

import torch
import clip
import math
import numpy as np
import pandas as pd

class CLIP():
    def __init__(self, model_name='ViT-B/32'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.model, self.preprocessor = clip.load(self.model_name, device=self.device)

    def encode_images(self, stacked_images: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            # Encode the images batch to compute the feature vectors and normalize them
            images_features = self.model.encode_image(stacked_images)
            images_features /= images_features.norm(dim=-1, keepdim=True)

        # Transfer the feature vectors back to the CPU and convert to numpy
        return images_features.cpu().numpy() 

    def encode_text_query(self, query:str) -> np.ndarray:
        with torch.no_grad():
            # Encode and normalize the description using CLIP
            text_encoded = self.model.encode_text(clip.tokenize(query).to(self.device))
            text_encoded /= text_encoded.norm(dim=-1, keepdim=True)
        
        return text_encoded

class dataset():
    def __init__(self, dataset_name=DATASET_NAME, src_path='', feature_path='', batch_size=16):
        self.src_path = src_path
        self.feature_path = feature_path
        self.clip_model = CLIP()
        self.image_paths = None
        self.image_names = None
        self.batch_size = batch_size
        self.features = None
        self.image_ids = None
        if dataset_name == 'Flicker-8k':
            self.extension = osp.join('Images', '*.jpg')
        elif dataset_name == 'V3C1':
            self.extension = '*.png'
        else:
            # Insert extension of other dataset to here
            pass

    def get_file_name(self):
        self.image_paths = sorted(glob(osp.join(self.src_path, self.extension)))
        # self.image_names = [convert_to_concepts(image_path)['filename'] for image_path in self.image_paths]

    def find_dominant_colors(self, image_batch: List[str], cluster=5) -> defaultdict:
        image_name = [convert_to_concepts(image_path)['filename'] for image_path in image_batch]
        dominant_colors_dict = defaultdict(list)
        for idx, image_file in enumerate(image_batch):
            img = cv2.imread(image_file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            color = DominantColors(img, cluster=cluster).dominant_colors()
            dominant_colors_dict[image_name[idx]] = color
        return dominant_colors_dict

    def compute_clip_features(self, image_batch: List[str]) -> np.ndarray:
        '''
        With the model and preprocessor were loaded from the CLIP pretrained-model
        '''
        # Load all the images from the files
        images = [Image.open(image_file) for image_file in image_batch]

        # Preprocess all images
        images_preprocessed = torch.stack([self.clip_model.preprocessor(image) for image in images]).to(self.clip_model.device)

        image_features = self.clip_model.encode_images(images_preprocessed)
        return image_features

    def preprocess_dataset(self, entire_dataset=True):
        '''
        Preprocessing dataset:
            - Load all images
            - comp
        '''
        if not self.image_paths:
            self.get_file_name()
        
        # Compute how many batches are needed
        if entire_dataset:
            print('Preprocess the whole dataset...')
            batches = math.ceil(len(self.image_paths) / self.batch_size)
        else:
            print('Preprocess a subset of the dataset...')
            batches = 10
        
        # Process each batch
        for i in tqdm(range(batches)):
            batch_ids_path = osp.join(self.feature_path, "ids", f"{i:010d}.csv")
            batch_features_path = osp.join(self.feature_path, "features", f"{i:010d}.npy")
        
            # Only do the processing if the batch wasn't processed yet
            if not osp.isdir(batch_features_path):
                # try:
                # Select the images for the current batch
                batch_files = self.image_paths[i*self.batch_size : (i+1)*self.batch_size]
                # Compute the features and save to a numpy file
                # batch_features = self.compute_clip_features(batch_files)
                # np.save(batch_features_path, batch_features)

                # Save the image IDs to a CSV file
                # image_ids = [image_file.split("/")[-1] for image_file in batch_files]
                # image_ids_data = pd.DataFrame(image_ids, columns=['image_id'])
                # image_ids_data.to_csv(batch_ids_path, index=False)

                image_colors = self.find_dominant_colors(batch_files)
                # print(image_colors)
                # except:
                #     # Catch problems with the processing to make the process more robust
                #     print(f'Problem with batch {i}')

    def load_dataset(self):
        '''
        Load saved metadata after preprocessing
        '''
        try:
            features_list = [np.load(feature_file) for feature_file in sorted(glob(osp.join(self.feature_path, "features", "*.npy")))]
            image_ids_list = [pd.read_csv(id_file) for id_file in sorted(glob(osp.join(self.feature_path, "ids", "*.csv")))]
            self.features = np.concatenate(features_list)
            self.image_ids = pd.concat(image_ids_list)
        except:
            print("There is no existing feature files.")

    def search_query(self, query: str, num_matches=500) -> List:
        if not self.features:
            self.load_dataset()

        text_encoded = self.clip_model.encode_text_query(query)
    
        # Retrieve the description vector and the image vectors
        text_features = text_encoded.cpu().numpy()

        # Compute the similarity between the descrption and each image using the Cosine similarity
        similarities = list((text_features @ self.features.T).squeeze(0))

        # Sort the images by their similarity score
        best_matched_images = sorted(zip(similarities, range(self.features.shape[0])), key=lambda x: x[0], reverse=True)
        return best_matched_images[:num_matches]

    
    def display_results(self, image_list=None):
        '''
        Display images from the top most matches list
        '''
        if image_list:
            try:
                image_ids = [self.image_paths[item[1]] for item in image_list]
                plot_figure(image_ids)
            except:
                print("Can't find best matched images.")


image_files = glob(osp.join(DATASET_PATH, "*.png"))
# print(image_files)
data = dataset(src_path=DATASET_PATH, feature_path=FEATURE_PATH)
data.get_file_name()
# print(data.image_paths[:10])
# print(data.image_names[:10])
# print(len(data.image_paths))
data.preprocess_dataset(entire_dataset=False)
# data.load_dataset()
# print("Features: ", len(data.features))

# query = "two people"
# best_images = data.search_query(query, num_matches=10)
# print("Length of features: ", len(data.features))
# print("Best images: ", best_images)
# data.compute_clip_features(image_batch=data.image_names[:10])
# print(len(data.image_features))