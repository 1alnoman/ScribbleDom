import scanpy as sc
import numpy as np
import pandas as pd

def find_neighbouring_clusters(sample,barcode_grid,mclust_result,i_row,i_col):
    neighbours_clusters = []

    if barcode_grid[i_row,i_col] is None:
        return neighbours_clusters
    
    neighbours = [[i_row,i_col-2],[i_row,i_col+2],[i_row-1,i_col-1],
    [i_row-1,i_col+1],[i_row+1,i_col-1],[i_row+1,i_col+1]]

    if sample == 'Melanoma':
        neighbours = [[i_row,i_col-1],[i_row,i_col+1],[i_row-1,i_col],[i_row+1,i_col]]

    for x,y in neighbours:
        if x<0 or y<0 or x>=barcode_grid.shape[0] or y>=barcode_grid.shape[1]:
            continue
        if barcode_grid[x,y] is None:
            continue
        neighbours_clusters.append(mclust_result.iloc[:,-1].loc[barcode_grid[x,y]])
    return neighbours_clusters

def make_backbone(preprocessed_data_folder, sample, dataset, threshold = 1, technology='visium'):
    cnt_file_path = f'./{preprocessed_data_folder}/{dataset}/{sample}/reading_h5/'
    mclust_result_csv = f'./{preprocessed_data_folder}/{dataset}/{sample}/mclust_result.csv'

    arr_row,arr_col = None,None
    if technology == 'visium':
        adata = sc.read_visium(cnt_file_path, count_file=f'{sample}_filtered_feature_bc_matrix.h5')
        arr_row,arr_col = adata.obs['array_row'],adata.obs['array_col']
    else:
        df_coord = pd.read_csv(f"./{preprocessed_data_folder}/{dataset}/{sample}/Coordinates/coordinates.csv",index_col=0)
        arr_row,arr_col = df_coord.iloc[:,-1],df_coord.iloc[:,-2]

    # define 2d grid of string as barcode
    barcode_grid = np.empty((arr_row.max()+1,arr_col.max()+1),dtype=object)
    barcode_grid[arr_row,arr_col] = arr_row.index

    mclust_result = pd.read_csv(mclust_result_csv,index_col=0)

    mclust_backbone = pd.DataFrame(index=mclust_result.index)
    mclust_backbone["cluster.init"] = None

    for i in range(barcode_grid.shape[0]):
        for j in range(barcode_grid.shape[1]):
            if barcode_grid[i,j] is not None:
                cluster_this = mclust_result.iloc[:,-1].loc[barcode_grid[i,j]]
                neighbours_clusters = find_neighbouring_clusters(sample,barcode_grid,mclust_result,i,j)

                if len(neighbours_clusters) == 0:
                    continue
                else :
                    correct_prop = np.where(np.array(neighbours_clusters)==cluster_this)[0].shape[0]/len(neighbours_clusters)
                    if correct_prop >= threshold:
                        mclust_backbone.iloc[:,-1].loc[barcode_grid[i,j]] = cluster_this

    mclust_backbone.to_csv(f'./{preprocessed_data_folder}/{dataset}/{sample}/mclust_backbone.csv')