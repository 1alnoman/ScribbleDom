import json
import argparse
from sklearn.metrics import adjusted_rand_score
import os
import pandas as pd

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
scheme = params['schema']

for sample in samples:
    manual_annotation_file = f"{preprocessed_dataset_folder}/{dataset}/{sample}/manual_annotations.csv"
    final_output_file = f"{final_output_folder}/{dataset}/{sample}/{scheme}/final_barcode_labels.csv"

    if not os.path.isfile(manual_annotation_file):
        print("Manual annnotaion doesn't exist!!!!")
        exit()
    if not os.path.isfile(final_output_file):
        print("Final output doesn't exist. Run the model to get final output!!!")

    df_man = pd.read_csv(manual_annotation_file,index_col=0)
    df_final = pd.read_csv(final_output_file,index_col=0)

    df_man.sort_index(inplace=True)
    df_final.sort_index(inplace=True)

    print(f"Ari for dataset:{dataset} sample:{sample} scheme:{scheme} is: ",adjusted_rand_score(df_man.iloc[:,-1],df_final.iloc[:,-1]))
