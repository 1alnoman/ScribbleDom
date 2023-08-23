#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import math
import anndata
import scipy
import scanpy as sc
import os

# %%
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import os
import math
import scanpy
import json
import argparse
from code_utils.scribble_generation.autoscribble_generator import make_backbone

parser = argparse.ArgumentParser(description='ScribbleSeg Preprocessor')
parser.add_argument('--params', help="The input parameters json file path", required=True)

args = parser.parse_args()

with open(args.params) as f:
   params = json.load(f)
dataset = params['dataset']
n_pcs = params['n_pcs']
samples = params['samples']
scheme = params['schema']
matrix_format_representation_of_data_path = params['matrix_represenation_of_ST_data_folder']
preprocessed_dataset_folder = params['preprocessed_data_folder']
technology = params['technology']


# # File Paths

# ### Input files path

# In[3]:

manual_scribble_filename = 'manual_scribble'

for sample in samples:
    mclust_scribbles_file_csv = f"{preprocessed_dataset_folder}/{dataset}/{sample}/mclust_result.csv"
    manual_scribble_file_csv = f"{preprocessed_dataset_folder}/{dataset}/{sample}/{manual_scribble_filename}.csv"
    mclust_backbone_file_csv = f"{preprocessed_dataset_folder}/{dataset}/{sample}/mclust_backbone.csv"
    pc_csv_path = f'./{preprocessed_dataset_folder}/{dataset}/{sample}/Principal_Components/CSV/pcs_{n_pcs}_from_bayesSpace_top_2000_HVGs.csv'

    map_pixel_to_grid_spot_file_path = f"{matrix_format_representation_of_data_path}/{dataset}/{sample}/Jsons/map_pixel_to_grid_spot.json"
    background_path = f"{matrix_format_representation_of_data_path}/{dataset}/{sample}/Npys/backgrounds.npy"
    foreground_path = f"{matrix_format_representation_of_data_path}/{dataset}/{sample}/Npys/foregrounds.npy"
    pixel_barcode_file_path = f"{matrix_format_representation_of_data_path}/{dataset}/{sample}/Npys/pixel_barcode.npy"
    pca_file_path = f"{matrix_format_representation_of_data_path}/{dataset}/{sample}/Npys/mapped_{n_pcs}.npy"
    mclust_scribbles_file = f"{matrix_format_representation_of_data_path}/{dataset}/{sample}/Scribble/mclust_scribble.npy"
    manual_scribbles_file = f"{matrix_format_representation_of_data_path}/{dataset}/{sample}/Scribble/manual_scribble.npy"
    mclust_backbone_file = f"{matrix_format_representation_of_data_path}/{dataset}/{sample}/Scribble/mclust_backbone_scribble.npy"

    def make_directory_if_not_exist(path):
        if not os.path.exists(path):
            os.makedirs(path)

    make_directory_if_not_exist(f'{matrix_format_representation_of_data_path}/{dataset}/{sample}/Jsons')
    make_directory_if_not_exist(f'{matrix_format_representation_of_data_path}/{dataset}/{sample}/Npys')
    make_directory_if_not_exist(f'{matrix_format_representation_of_data_path}/{dataset}/{sample}/Scribble')

    make_backbone(preprocessed_dataset_folder,sample,dataset,threshold=1,technology=technology)

    # In[5]:


    os.makedirs(os.path.dirname(map_pixel_to_grid_spot_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(background_path), exist_ok=True)
    os.makedirs(os.path.dirname(foreground_path), exist_ok=True)
    os.makedirs(os.path.dirname(pixel_barcode_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(pca_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(mclust_scribbles_file_csv), exist_ok=True)
    if scheme == 'expert':
        os.makedirs(os.path.dirname(manual_scribble_file_csv), exist_ok=True)
    os.makedirs(os.path.dirname(mclust_backbone_file_csv), exist_ok=True)


    df_coord = pd.read_csv(f"./{preprocessed_dataset_folder}/{dataset}/{sample}/Coordinates/coordinates.csv",index_col=0)
    x_pixels = []
    y_pixels = []
    labels = []
    for i in range(df_coord.shape[0]):
        spot = df_coord.index[i]
        x_pixels.append(int(spot.split("x")[1])-1)
        y_pixels.append(int(spot.split("x")[0])+1)


    # In[13]:

    pcs_from_BayesSpace = pd.read_csv(pc_csv_path, index_col=0)
    pcs_7 = pcs_from_BayesSpace.values
    pcs_7.shape


    mapped_7 = np.zeros((max(y_pixels)+1, max(x_pixels)+1, 7))
    for i in range(len(x_pixels)):
        mapped_7[y_pixels[i], x_pixels[i]] = pcs_7[i]
    np.save(f"{matrix_format_representation_of_data_path}/{dataset}/{sample}/Npys/mapped_{n_pcs}.npy", mapped_7)


    # In[16]:


    def make_grid_idx(x_pixels,y_pixels):
        grid_idx = np.zeros((max(y_pixels)+1,max(x_pixels)+1)) - 1
        for i in range(len(x_pixels)):
            grid_idx[y_pixels[i],x_pixels[i]] = 1
        return grid_idx

    def get_pixel_to_grid_spot_map(grid_pixel_coor, grid_idx):
        n = grid_pixel_coor.shape[0]
        m = grid_pixel_coor.shape[1]
        map_pixel_to_grid_spot = {}
        for i in range(n):
            for j in range(m):
                if grid_idx[i, j] != -1:
                    map_from = f'({i}, {j})'
                    map_to = (i, j)
                    map_pixel_to_grid_spot[map_from] = map_to
        return map_pixel_to_grid_spot


    # In[17]:


    grid_idx = make_grid_idx(x_pixels,y_pixels)
    map_pixel_to_grid_spot = get_pixel_to_grid_spot_map(grid_idx, grid_idx)



    with open(map_pixel_to_grid_spot_file_path, "w") as outfile:
        json.dump(map_pixel_to_grid_spot, outfile)


    # In[19]:


    backgrounds = np.argwhere(grid_idx == -1)
    foregrounds = np.argwhere(grid_idx == 1)


    # In[20]:


    np.save(background_path, backgrounds)
    np.save(foreground_path, foregrounds)


    # In[21]:


    def make_pixel_barcode(grid_idx,x_pixels,y_pixels):
        mx_row,mx_col = grid_idx.shape

        pixel_barcode = np.empty([mx_row + 1, mx_col + 1], dtype='<U100')

        for i in range(len(x_pixels)):
            pixel_barcode[y_pixels[i], x_pixels[i]] = str(y_pixels[i]-1) + "x" + str(x_pixels[i]+1)
        return pixel_barcode


    # In[22]:


    pixel_barcode = make_pixel_barcode(grid_idx,x_pixels,y_pixels)
    np.save(pixel_barcode_file_path, pixel_barcode)


    # In[23]:


    new_scribble_spot = np.full_like(grid_idx, 255)
    new_backbone_spot = np.full_like(grid_idx, 255)

    df_mclust_scribble = pd.read_csv(mclust_scribbles_file_csv,index_col=0)
    df_mclust_scribble.head()


    # In[24]:


    x_pixels = []
    y_pixels = []
    labels = []
    for i in range(df_mclust_scribble.shape[0]):
        spot = df_mclust_scribble.index[i]
        x_pixels.append(int(spot.split("x")[1])-1)
        y_pixels.append(int(spot.split("x")[0])+1)
        labels.append(df_mclust_scribble.iloc[i,0])



    # In[25]:


    new_scribble_spot[y_pixels,x_pixels] = labels
    np.save(mclust_scribbles_file,new_scribble_spot)


    new_scribble_spot = np.full_like(grid_idx, 255)

    if scheme == 'expert':
        df_manual_scribble = pd.read_csv(manual_scribble_file_csv,index_col=0)
    df_mclust_backbone = pd.read_csv(mclust_backbone_file_csv,index_col=0)


    if scheme == 'expert':
        x_pixels = []
        y_pixels = []
        x_pixels_col = []
        y_pixels_col = []
        labels_scribble = []
        labels_scribble_col = []
        for i in range(df_manual_scribble.shape[0]):
            spot = df_manual_scribble.index[i]
            if not math.isnan(df_manual_scribble.iloc[i,0]):
                x_pixels.append(int(spot.split("x")[1])-1)
                y_pixels.append(int(spot.split("x")[0])+1)
                x_pixels_col.append(int(spot.split("x")[1])-1)
                y_pixels_col.append(int(spot.split("x")[0])+1)
                labels_scribble.append(df_manual_scribble.iloc[i,0]+1)
                labels_scribble_col.append(df_manual_scribble.iloc[i,0]+1)
            else:
                x_pixels_col.append(int(spot.split("x")[1])-1)
                y_pixels_col.append(int(spot.split("x")[0])+1)
                labels_scribble_col.append(-1)
        new_scribble_spot[y_pixels,x_pixels] = labels_scribble
        np.save(manual_scribbles_file,new_scribble_spot)


    x_pixels = []
    y_pixels = []
    labels_backbone = []
    for i in range(df_mclust_backbone.shape[0]):
        spot = df_mclust_backbone.index[i]
        if not math.isnan(df_mclust_backbone.iloc[i,0]):
            x_pixels.append(int(spot.split("x")[1])-1)
            y_pixels.append(int(spot.split("x")[0])+1)
            labels_backbone.append(df_mclust_backbone.iloc[i,0]+1)


    # In[33]:


    new_backbone_spot[y_pixels,x_pixels] = labels_backbone
    np.save(mclust_backbone_file,new_backbone_spot)


    # In[ ]:




