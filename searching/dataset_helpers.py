from configs import *
from helpers import *
from PIL import Image
from typing import List
from collections import defaultdict

import torch
import clip
import math
import joblib
import numpy as np
import pandas as pd

# Define the CLIP encoding model class
class CLIP():
    def __init__(self, model_name='ViT-B/32'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.model, self.encoder = clip.load(self.model_name, device=self.device)

    def encode_images(self, stacked_images: torch.Tensor) -> np.ndarray:
        '''
        Function to transform images in the dataset into feature vectors
        
        params:
            - stacked_images: Tensor
                A stack of images to encode

        return:
            - List of feature vectors of the images
        ''' 
        with torch.no_grad():
            # Encode the images batch to compute the feature vectors and normalize them
            images_features = self.model.encode_image(stacked_images)
            images_features /= images_features.norm(dim=-1, keepdim=True)

        # Transfer the feature vectors back to the CPU and convert to numpy
        return images_features.cpu().numpy() 

    def encode_text_query(self, query:str) -> np.ndarray:
        '''
        Function to transform a text query string into vector
        
        params:
            - query: string
                An input string to encode
        
        return:
            - A numerical array of encoded text string
        '''
        with torch.no_grad():
            # Encode and normalize the description using CLIP
            text_encoded = self.model.encode_text(clip.tokenize(query).to(self.device))
            text_encoded /= text_encoded.norm(dim=-1, keepdim=True)
        
        return text_encoded

class dataset():
    def __init__(self, dataset_name=DATASET_NAME, src_path='', feature_path='', batch_size=16, generate_features=False):
        self.src_path = src_path
        self.feature_path = feature_path
        self.clip_model = CLIP()
        self.image_names = None
        self.batch_size = batch_size
        self.features = {}
        self.generate_features = generate_features
        # self.image_ids = None
        if dataset_name == 'Flicker-8k':
            self.extension = osp.join('Images', '*.jpg')
        elif dataset_name == 'V3C1':
            self.extension = '*.png'
        else:
            # Insert extension of other dataset to here
            pass

    def get_file_name(self):
        '''
        Function to get a list of images' names from the source path in ascending order
        '''
        self.image_names = sorted(glob(osp.join(self.src_path, self.extension)))

    def compute_clip_image_embeddings(self, image_batch: List[str]) -> np.ndarray:
        '''
        Encoded image list into vectors using the model and encoder loaded from CLIP
        
        params:
           - image_batch: List(str)
                A batch of images to encode
        
        return:
           - image_embeddings: np.ndarray
                image embedding vectors
        '''
        # Sort the file name of all images in batch by
        image_batch = sorted(image_batch)
        image_embeddings_dict = defaultdict(list)

        # Load all the images from the files
        images = [Image.open(image_file) for image_file in image_batch]
        filenames = [convert_to_concepts(image_file)['filename'] for image_file in image_batch]
        
        # Encode all images
        images_encoded = torch.stack([self.clip_model.encoder(image) for image in images]).to(self.clip_model.device)
        image_embeddings = self.clip_model.encode_images(images_encoded)

        # Match file name with the embedding vectors
        for idx in range(len(image_batch)):
            image_embeddings_dict[filenames[idx]] = image_embeddings[idx]

        return image_embeddings_dict

    def encode_dataset(self, entire_dataset=True):
        '''
        Images will be divided into batches and encoded into embedding vectors.
        Then all embedding files will be saved for later use.
        
        params:
            - entire_dataset: bool, default=True
                Whether process the entire dataset or not

        returns:
            - defaultdict
                Dictionary with keys are the image names and values are the embedding vectors
        '''
        
        if self.image_names is None:
            self.get_file_name()
        
        # Compute how many batches are needed
        if entire_dataset:
            print('Encode the whole dataset...')
            batches = math.ceil(len(self.image_names) / self.batch_size)
        else:
            print('Encode a subset of the dataset...')
            batches = 10
        
        # Process each batch
        for i in tqdm(range(batches)):
        # for i in tqdm(range(10)):
            embedding_filename = osp.join(self.feature_path, f'{i:010d}.joblib')

            # Only do the processing if the batch wasn't processed yet
            # if not osp.isdir(embedding_filename):
            if self.generate_features:
                try:
                    # Select the images for the current batch
                    batch_files = self.image_names[i*self.batch_size : (i+1)*self.batch_size]

                    # Compute the features and save to a joblib file
                    batch_embeddings = self.compute_clip_image_embeddings(batch_files)
                    joblib.dump(batch_embeddings, embedding_filename)

                except:
                    # Catch problems with the processing to make the process more robust
                    print(f'Problem with batch {i}')


    def load_dataset(self):
        '''
        Load saved metadata (extracted features) after encoding
        '''
        try:
            features_list = [joblib.load(feature_file) for feature_file in sorted(glob(osp.join(self.feature_path, '*.joblib')))]
            for item in features_list:
                self.features.update(item)
            # features_list = [np.load(feature_file) for feature_file in sorted(glob(osp.join(self.feature_path, 'features', '*.npy')))]
            # image_ids_list = [pd.read_csv(id_file) for id_file in sorted(glob(osp.join(self.feature_path, 'ids', '*.csv')))]
            # self.features = np.concatenate(features_list)
            # self.image_ids = pd.concat(image_ids_list)
        except:
            print('There is no existing feature files.')

    def search_query(self, query: str, num_matches=500) -> List:
        '''
        Function to search for target images giving an input query
        
        params:
            - query: string
                An input text query to search for the target images
            - num_matches: integer, default=500
                The number of images matching the query
        
        return:
            - A list of matching images to the input query
        '''
        if self.features is None:
            self.load_dataset()

        text_encoded = self.clip_model.encode_text_query(query)
    
        # Retrieve the description vector and the image vectors
        text_features = text_encoded.cpu().numpy()

        # Compute the similarity between the descrption and each image using the Cosine similarity
        similarities = list((text_features @ self.features.T).squeeze(0))

        # Sort the images by their similarity score
        best_matched_images = sorted(zip(similarities, range(self.features.shape[0])), key=lambda x: x[0], reverse=True)
        best_matched_image_names = [self.image_names[item[1]] for item in best_matched_images]
        result = [convert_to_concepts(item) for item in best_matched_image_names[:num_matches]]
        return result

    
    def display_results(self, image_list=None, subplot_size=(5, 3)):
        '''
        Display images from the top most matches list
        '''
        if image_list:
            try:
                image_ids = [osp.join(self.src_path, item['filename']) for item in image_list]
                plot_figure(image_ids, subplot_size=subplot_size)
            except:
                print('Can\'t find best matched images.')


# images_files = glob(osp.join(DATASET_PATH, 'Images', '*.jpg'))
# data = dataset(src_path=DATASET_PATH, feature_path=FEATURE_PATH)
# data.get_file_name()
# # data.encode_dataset(entire_dataset=False)
# # data.load_dataset()
# print('Features: ', data.features)

# query = 'two people'
# best_images = data.search_query(query, num_matches=10)
# print('Length of features: ', len(data.features))
# print('Best images: ', best_images)
# data.compute_clip_image_embeddings(image_batch=data.image_names[:10])
# print(len(data.image_features))