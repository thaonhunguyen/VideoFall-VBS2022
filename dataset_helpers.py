from configs import *
from helpers import *
from PIL import Image
from typing import List
from collections import defaultdict

import torch
import clip
import math
import joblib
import faiss
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
    def __init__(self, dataset_name=DATASET_NAME, src_path=''):
        self.src_path = src_path
        self.image_names = None
        if dataset_name == 'Flicker-8k':
            self.extension = osp.join('Images', '*.jpg')
        elif dataset_name == 'LSC':
            self.extension = osp.join('*', '*.jpg')
        else:
            self.extension = '*.png'
            pass

    def get_file_name(self, load_file=True):
        '''
        Function to get a list of images' names from the source path in ascending order
        
        params:
            - load_file: bool, default=True
                Whether load the saved file of all image names or not
        '''
        if load_file==True:
            print("Loading all image names ...")
            self.image_names = joblib.load(IMAGE_NAME_PATH)
        else:
            print("Getting all image names from the source path ...")
            self.image_names = []
            folder_names = sort_list(os.listdir(DATASET_PATH))
            for folder in tqdm(folder_names):
                folder_path = osp.join(DATASET_PATH, folder)
                filenames = sort_list(os.listdir(folder_path))
                self.image_names.extend([osp.join(folder_path, filename) for filename in filenames])
                
            if dataset_name == 'LSC':
                error_image_list = joblib.load(error_image_file)
                for image in error_image_list:
                    self.image_names.remove(image)

class CLIPSearchEngine():
    def __init__(self, dataset_name=DATASET_NAME, src_path='', feature_path='', batch_size=16, generate_features=False):
        self.dataset_name = dataset_name
        self.dataset = dataset(dataset_name=self.dataset_name, src_path=src_path)
        self.clip_model = CLIP()
        self.feature_dict = {}
        self.features = None
        self.feature_path = feature_path
        self.batch_size = batch_size
        self.generate_features = generate_features

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
        image_batch = sort_list(image_batch)
        image_embeddings_dict = defaultdict(list)
        # Load all the images from the files          
        images = [Image.open(image_file) for image_file in image_batch]
        filenames = [convert_to_concepts(image_file, dataset_name=self.dataset_name)['filename'] for image_file in image_batch]
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
        
        if self.dataset.image_names is None:
            self.dataset.get_file_name()
        
        # Compute how many batches are needed
        if entire_dataset:
            print('Encode the whole dataset...')
            batches = math.ceil(len(self.dataset.image_names) / self.batch_size)
        else:
            print('Encode a subset of the dataset...')
            batches = 10
        
        if self.generate_features:
            print("Generate features ...")
            # Process each batch
            for i in tqdm(range(batches)):
            # for i in tqdm(range(10)):
                embedding_filename = osp.join(self.feature_path, f'{i:010d}.joblib')
                try:
                    # Select the images for the current batch
                    batch_files = self.dataset.image_names[i*self.batch_size : (i+1)*self.batch_size]
                    # Compute the features and save to a joblib file
                    batch_embeddings = self.compute_clip_image_embeddings(batch_files)
                    joblib.dump(batch_embeddings, embedding_filename)
                except:
                    print(f"Problem with batch {i}.")
        else:
            print("Load extracted features ...")
            self.load_features()

    def load_features(self):
        '''
        Load saved metadata (extracted features) after encoding
        '''
        try:
            print("Loading feature files ...")
            feature_list  = sort_list(glob(osp.join(self.feature_path, '*.joblib')))
            for feature_file in tqdm(feature_list):
                feature = joblib.load(feature_file)
                self.feature_dict.update(feature)
                del feature
            temp = self.feature_dict.values()
            self.features = np.asarray([*temp]).astype('float32')
            del temp
        except:
            print('There is no existing feature files.')
            
    def encode_input_query(self, query: str) -> np.array:
        '''
        Function to encode an input query into feature vector
        
        params:
            - query: str
                An input text query to search for the target images
                
        return:
            - feature_vector: array
                An embedded feature vector with shape (len, 1)
        '''
        if is_image(query):
            img_query = convert_to_concepts(query, dataset_name=DATASET_NAME)['filename']
            feature = self.feature_dict[img_query]
            feature_vec = np.expand_dims(feature, axis=0)
            feature_vector = feature_vec.astype('float32')
        else:
            # Encode the string query into the latent space
            str_feature = self.clip_model.encode_text_query(query)
            feature_vector = str_feature.cpu().numpy().astype('float32')
        
        return feature_vector
        
    def search_query(self, query: str, num_matches=500, nlist=10, ss_type='faiss') -> List:
        '''
        Function to search for target images giving an input query
        
        params:
            - query: str
                An input text query to search for the target images
            - num_matches: integer, default=500
                The number of images matching the query
        
        return:
            - A list of matching images to the input query
        '''
        if self.features is None:
            self.load_features()

        # Encode the input query into feature vector
        feature_vector = self.encode_input_query(query)
        
        if ss_type == 'faiss':
            dimension = self.features.shape[1]
            # Initialize faiss searching object
            quantiser = faiss.IndexFlatL2(dimension)  
            index = faiss.IndexIVFFlat(quantiser, dimension, nlist, faiss.METRIC_L2)
            index.train(self.features) 
            index.add(self.features)  
            # Calculate the distances of the text vectors 
            distances, indices = index.search(feature_vector, num_matches)
            best_matched_image_names = [self.dataset.image_names[item] for item in indices[0]]
        else:
            # Compute the similarity between the description and each image using the Cosine similarity
            similarities = list((feature_vector @ self.features.T).squeeze(0))
            # Sort the images by their similarity scores
            indices = sorted(zip(similarities, range(self.features.shape[0])), key=lambda x: x[0], reverse=True)
            best_matched_image_names = [self.dataset.image_names[item[1]] for item in indices]
            
        result = [convert_to_concepts(item, dataset_name=self.dataset_name) for item in best_matched_image_names[:num_matches]]
        return result
    
    def display_results(self, image_list=None, subplot_size=(5, 3)):
        '''
        Visualize images from the top most similar image list
        
        params:
            - image_list: List, default=None
                An input image list to display
            - subplot_size: tuple, default=(5, 3)
                The size of the plot to visualize
        '''
        if image_list:
            try:
                image_ids = [osp.join(self.dataset.src_path, item['filename']) for item in image_list]
                plot_figures(image_ids, subplot_size=subplot_size)
            except:
                print('Can\'t find best matched images.')

# images_files = glob(osp.join(DATASET_PATH, 'Images', '*.jpg'))
# data = dataset(src_path=DATASET_PATH, feature_path=FEATURE_PATH)
# data.get_file_name()
# # data.encode_dataset(entire_dataset=False)
# # data.load_features()
# print('Features: ', data.features)

# query = 'two people'
# best_images = data.search_query(query, num_matches=10)
# print('Length of features: ', len(data.features))
# print('Best images: ', best_images)
# data.compute_clip_image_embeddings(image_batch=data.image_names[:10])
# print(len(data.image_features))