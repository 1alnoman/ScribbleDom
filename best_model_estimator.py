import pandas as pd
import numpy as np
import glob
from scipy.stats import multivariate_normal
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score
import shutil
import os
 
import json
import argparse

parser = argparse.ArgumentParser(description='Likelikehood estimation of models')
parser.add_argument('--params', help="The input of model_outs and preprocessed_data and location of output", required=True)
args = parser.parse_args()

with open(args.params) as f:
   params = json.load(f)

model_outputs_folder = params['model_output_folder']
preprocessed_dataset_folder = params['preprocessed_data_folder']
final_output_folder = params['final_output_folder']
dataset = params['dataset']
samples = params['samples']
npcs = params['n_pcs']

def make_directory_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def NN_graph_index(X, NN = 6):
    # X = coordinates, size N x D
    # NN = number of nearest neighbors
    kt = KDTree(X)
    G_idx = kt.query(X, NN+1)[1][:, 1:]
    return G_idx

def calculate_p_D(labels, G_idx, gamma=3, NN = 6):
    def hist_1d(a):
        return np.histogram(a, bins=label_bins)[0]
    
    r = labels[G_idx.ravel()].reshape(-1, NN)
    label_bins = np.arange(df["label"].unique().shape[0] + 5)
    counts = np.apply_along_axis(hist_1d, axis=1, arr=r)
    p = np.exp(gamma*counts/NN)
    norm_fact = p.sum(axis=1).reshape(-1, 1)
    p /= norm_fact
    p_D = p[np.arange(len(p)), labels]
    return p_D


def calculate_p_y_given_D(x):
    mu = x.mean(axis=0)
    Sigma = np.cov(x, rowvar=False, bias=True)
    p_y_given_D = multivariate_normal.pdf(x, mean=mu, cov=Sigma,allow_singular=True)
    return p_y_given_D

def get_Human_DLPFC_filenames(dataset, sample, fname_out_root):
    fname = {}
    fname["data"] = f"{preprocessed_dataset_folder}/{dataset}/{sample}/Principal_Components/CSV/pcs_{npcs}_from_bayesSpace_top_2000_HVGs.csv"
    fname["spatial"] = f"{preprocessed_dataset_folder}/{dataset}/{sample}/Coordinates/coordinates.csv"
    fname["labels"] = np.sort(glob.glob(fname_out_root + "final_barcode_labels.csv"))
    fname["meta"] = np.sort(glob.glob(fname_out_root + "meta_data.csv"))
    return fname

df_results = {}
var_results = {}

for sample in samples:
    fname_out_root = f"./{model_outputs_folder}/{dataset}/{sample}/*/" 
    fname = get_Human_DLPFC_filenames(dataset, sample, fname_out_root)
    
    df = pd.read_csv(fname["data"], index_col=0)
    X = pd.read_csv(fname["spatial"], index_col = 0).values
    Graph_index = NN_graph_index(X)
    
    df_result = pd.DataFrame({})

    var_results[sample] = {}
    for i in range(len(fname["labels"])):
        df_label = pd.read_csv(fname["labels"][i], index_col=0)
        df["label"] = df_label.loc[df.index]
        
        df_meta = pd.read_csv(fname["meta"][i], index_col=0)

        # calculating spatial probability 
        df["p_D"] = calculate_p_D(df["label"].values, Graph_index)

        ll_joint = 0
        ll_cond = 0

        hyper_name = fname["labels"][i].split("/")[-2]
        var_results[sample][hyper_name] = {}

        for c in df["label"].unique():
            df_c = df.groupby("label").get_group(c)
            p_D = df_c.iloc[:, -1].values
                                                
            x = df_c.iloc[:, :-2].values
            if x.shape[0] == 1: continue

            p_y_given_D = calculate_p_y_given_D(x)
            
            # test
            var_c = np.diag(np.cov(x, rowvar=False, bias=True)).sum()
            var_results[sample][hyper_name][c] = var_c
            
            p_joint = p_y_given_D*p_D
            
            
            ll_joint += np.log(p_joint).sum()
            ll_cond += np.log(p_y_given_D).sum()
            
        df_meta["log-likelihood_joint"] = ll_joint
        df_meta["log-likelihood_conditional"] = ll_cond
        df_meta["variance"] = sum(var_results[sample][hyper_name].values())


        df_result = pd.concat([df_result, df_meta], ignore_index=True)
        
    df_results[sample] = df_result

for sample in samples:
    y1 = df_results[sample]["log-likelihood_joint"]
    y2 = df_results[sample]["log-likelihood_conditional"]
    y3 = df_results[sample]["log-likelihood_joint"] - df_results[sample]["variance"]*100

    idx1 = np.argmax(y1)
    idx2 = np.argmax(y2)
    idx3 = np.argmax(y3)

    fname_out_root = f"./{model_outputs_folder}/{dataset}/{sample}/*/" 
    fname = get_Human_DLPFC_filenames(dataset, sample, fname_out_root)

    src_final_meta = fname['meta'][idx3]
    src_final_label = fname['labels'][idx3]

    final_output_dir = f"{final_output_folder}/{dataset}/{sample}"
    make_directory_if_not_exist(final_output_dir)

    dest_final_meta = f"{final_output_dir}/meta_data.csv"
    dest_final_label = f"{final_output_dir}/final_barcode_labels.csv"

    shutil.copyfile(src_final_meta,dest_final_meta)
    shutil.copyfile(src_final_label,dest_final_label)