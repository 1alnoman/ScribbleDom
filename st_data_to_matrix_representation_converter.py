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

for sample in samples:
    h5_path = f'./{preprocessed_dataset_folder}/{dataset}/{sample}/reading_h5/'
    h5_file = f'{sample}_filtered_feature_bc_matrix.h5'
    if dataset != 'Human_DLPFC':
        adata = scanpy.read(h5_path+h5_file)
    else :
        adata = scanpy.read_visium(path=h5_path,count_file=h5_file)
    adata.var_names_make_unique()
    
    # %%
    mapped_pc_file_path = f'./{matrix_format_representation_of_data_path}/{dataset}/{sample}/Npys/mapped_{n_pcs}.npy'
    backgrounds_file_path = f'{matrix_format_representation_of_data_path}/{dataset}/{sample}/Npys/backgrounds.npy'
    foregrounds_file_path = f'{matrix_format_representation_of_data_path}/{dataset}/{sample}/Npys/foregrounds.npy'
    pixel_barcode_file_path = f'{matrix_format_representation_of_data_path}/{dataset}/{sample}/Npys/pixel_barcode.npy'

    pc_csv_path = f'./{preprocessed_dataset_folder}/{dataset}/{sample}/Principal_Components/CSV/pcs_{n_pcs}_from_bayesSpace_top_2000_HVGs.csv'
    scr_csv_path = f'./{preprocessed_dataset_folder}/{dataset}/{sample}/manual_scribble.csv'
    scr_file_path = f'./{matrix_format_representation_of_data_path}/{dataset}/{sample}/Scribble/manual_scribble.npy'

    if scheme == 'mclust':
        make_backbone(preprocessed_data_folder=preprocessed_dataset_folder,sample=sample,dataset=dataset)
        scr_csv_path = f'./{preprocessed_dataset_folder}/{dataset}/{sample}/mclust_backbone.csv'
        scr_file_path = f'./{matrix_format_representation_of_data_path}/{dataset}/{sample}/Scribble/mclust_backbone_scribble.npy'

    def make_directory_if_not_exist(path):
        if not os.path.exists(path):
            os.makedirs(path)

    make_directory_if_not_exist(f'{matrix_format_representation_of_data_path}/{dataset}/{sample}/Npys')
    make_directory_if_not_exist(f'{matrix_format_representation_of_data_path}/{dataset}/{sample}/Scribble')

    # %%
    def make_grid_idx(adata):
        n = adata.obs['array_row'].max() + 1
        m = adata.obs['array_col'].max() + 1
        grid_idx = np.zeros((n, m), dtype='int') - 1
        spot_rows = adata.obs['array_row']
        spot_cols = adata.obs['array_col']
        grid_idx[spot_rows, spot_cols] = range(len(adata.obs.index))
        return grid_idx

    def make_grid_barcode(adata):
        n = adata.obs['array_row'].max() + 1
        m = adata.obs['array_col'].max() + 1
        grid_barcode = np.empty([n, m], dtype='<U100')
        spot_rows = adata.obs['array_row']
        spot_cols = adata.obs['array_col']
        grid_barcode[spot_rows, spot_cols] = adata.obs.index
        return grid_barcode

    # %%
    def check_grid_validity_return_starting_pos(grid):
        '''
        Check if the grid is valid or not, valid if (i + j) % 2 for all non -1s are equal where i and j can be row and col
        Returns (i + j) % 2 of any 1 present in grid
        '''
        n = grid.shape[0]
        m = grid.shape[1]
        parity = -1
        started = False
        for i in range(n):
            for j in range(m):
                if grid[i, j] != -1 and not started:
                    parity = (i + j) % 2
                    started = True
                if grid[i, j] != -1 and parity != (i + j) % 2:
                    print("Invalid grid structure!")
                    return -1
        return parity

    # %%
    def refine(grid):pass

    # %%
    grid_idx = make_grid_idx(adata)
    grid_barcode = make_grid_barcode(adata)
    parity = check_grid_validity_return_starting_pos(grid_idx)

    # %%
    def make_grid_pixel_coor(grid_idx, parity):

        n = grid_idx.shape[0]
        m = grid_idx.shape[1]

        grid_pixel_coor = np.zeros((n, m + 2, 2), dtype=int) - 1

        n = grid_pixel_coor.shape[0]
        m = grid_pixel_coor.shape[1]

        for i in range(n):
            for j in range(m):
                if (i + j) % 2 == parity:
                    if i == 0 and j <= 1:
                        grid_pixel_coor[0, j, :] = 0
                    else:
                        if j <= 1:
                            grid_pixel_coor[i, j, :] = grid_pixel_coor[i - 1, j + 1, :] + [1, 0]
                        else:
                            grid_pixel_coor[i, j, :] = grid_pixel_coor[i, j - 2, :] + [0, 1]
                    

        return grid_pixel_coor[:,:-2]

    # %%
    grid_pixel_coor = make_grid_pixel_coor(grid_idx, parity)
    # %%
    pixel_coor = grid_pixel_coor.reshape((-1, 2))

    # %%
    mx = pixel_coor[:, 0].max()
    color = grid_idx.flatten()

    # %%
    color[color != -1] = 1
    if dataset == 'Custom': rad = 2000
    else: rad = 10


    # %%
    def make_grid_pc(grid_pixel_coor, grid_barcode, map_barcode_pc):
        mx_row = grid_pixel_coor[:, :, 0].max()
        mx_col = grid_pixel_coor[:, :, 1].max()
        grid_pc = np.zeros((mx_row + 1, mx_col + 1, n_pcs))
        idx_rows_cols = np.argwhere(grid_barcode != '')
        idx_rows = idx_rows_cols[:, 0]
        idx_cols = idx_rows_cols[:, 1]
        pixel_rows_cols = grid_pixel_coor[idx_rows, idx_cols]
        barcode_sequence = grid_barcode[grid_barcode != '']
        grid_pc[pixel_rows_cols[:, 0], pixel_rows_cols[:, 1], :] = [map_barcode_pc[barcode] for barcode in barcode_sequence]
        return grid_pc

    # %%
    def make_grid_scr(grid_pixel_coor, grid_barcode, df_barcode_scr):
        mx_row = grid_pixel_coor[:, :, 0].max()
        mx_col = grid_pixel_coor[:, :, 1].max()
        grid_scr = np.zeros((mx_row + 1, mx_col + 1), dtype='int') + 255
        idx_rows_cols = np.argwhere(grid_barcode != '')
        idx_rows = idx_rows_cols[:, 0]
        idx_cols = idx_rows_cols[:, 1]
        pixel_rows_cols = grid_pixel_coor[idx_rows, idx_cols]
        barcode_sequence = grid_barcode[grid_barcode != '']
        grid_scr[pixel_rows_cols[:, 0], pixel_rows_cols[:, 1]] = [df_barcode_scr.iloc[:,-1][barcode] for barcode in barcode_sequence]
        return grid_scr

    # %%
    def make_pixel_barcode(grid_pixel_coor, grid_barcode):
        mx_row = grid_pixel_coor[:, :, 0].max()
        mx_col = grid_pixel_coor[:, :, 1].max()
        pixel_barcode = np.empty([mx_row + 1, mx_col + 1], dtype='<U100')
        idx_rows_cols = np.argwhere(grid_barcode != '')
        idx_rows = idx_rows_cols[:, 0]
        idx_cols = idx_rows_cols[:, 1]
        pixel_rows_cols = grid_pixel_coor[idx_rows, idx_cols]
        barcode_sequence = grid_barcode[grid_barcode != '']
        pixel_barcode[pixel_rows_cols[:, 0], pixel_rows_cols[:, 1]] = barcode_sequence
        return pixel_barcode

    # %%
    pixel_barcode = make_pixel_barcode(grid_pixel_coor, grid_barcode)
    np.save(pixel_barcode_file_path, pixel_barcode)

    # %%
    def make_map_barcode_pc(csv_path):
        df_pc = pd.read_csv(csv_path, index_col=0)
        map_barcode_pc = dict(zip(df_pc.index, df_pc.values))
        return map_barcode_pc

    # %%
    map_barcode_pc = make_map_barcode_pc(pc_csv_path)

    # %%
    grid_idx = make_grid_idx(adata)
    grid_barcode = make_grid_barcode(adata)
    parity = check_grid_validity_return_starting_pos(grid_idx)
    if parity == -1: refine(grid_idx)

    # grid_01 = make_grid01(grid_idx)
    grid_pixel_coor = make_grid_pixel_coor(grid_idx, parity)
    grid_pc = make_grid_pc(grid_pixel_coor, grid_barcode, map_barcode_pc)

    # %%
    df_barcode_scr = pd.read_csv(scr_csv_path, index_col=0).fillna(255).astype('int')
    grid_scr = make_grid_scr(grid_pixel_coor, grid_barcode, df_barcode_scr)

    np.save(scr_file_path, grid_scr)
    np.save(mapped_pc_file_path, grid_pc)

    # %%
    def find_backgrounds(grid_pixel_coor, grid_barcode):
        pixel_coor = grid_pixel_coor.reshape((-1, 2))
        n = pixel_coor[:, 0].max() + 1
        m = pixel_coor[:, 1].max() + 1
        grid_binary = np.zeros((n, m), dtype='int')
        idx_rows_cols = np.argwhere(grid_barcode != '')
        idx_rows = idx_rows_cols[:, 0]
        idx_cols = idx_rows_cols[:, 1]
        pixel_rows_cols = grid_pixel_coor[idx_rows, idx_cols]

        grid_binary[pixel_rows_cols[:, 0], pixel_rows_cols[:, 1]] = 1
        background = np.argwhere(grid_binary == 0)
        foreground = np.argwhere(grid_binary == 1)
        return background, foreground

    # %%
    background, foreground = find_backgrounds(grid_pixel_coor, grid_barcode)


    # %%
    np.save(backgrounds_file_path, background)
    np.save(foregrounds_file_path, foreground)
