import json
import argparse
from sklearn.metrics import adjusted_rand_score
import os
import pandas as pd
import matplotlib.pyplot as plt

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
    coordinate_file = f"{preprocessed_dataset_folder}/{dataset}/{sample}/Coordinates/coordinates.csv"
    final_output_file = f"{final_output_folder}/{dataset}/{sample}/{scheme}/final_barcode_labels.csv"
    final_output_img = f"{final_output_folder}/{dataset}/{sample}/{scheme}/final_out.png"
    final_output_ari = f"{final_output_folder}/{dataset}/{sample}/{scheme}/ari.csv"

    if not os.path.isfile(final_output_file):
        print("Final output doesn't exist. Run the model to get final output!!!")
        exit()

    df_final = pd.read_csv(final_output_file,index_col=0)
    df_coord = pd.read_csv(coordinate_file,index_col=0)
    df_final.sort_index(inplace=True)
    df_coord.sort_index(inplace=True)

    df_merged = pd.merge(df_final, df_coord, left_index=True, right_index=True)
    cols = df_merged.columns
    for col in cols:
        df_merged[col] = df_merged[col].values.astype('int')

    plot_color=["#F56867","#FEB915","#C798EE","#59BE86","#7495D3","#D1D1D1","#6D1A9C","#15821E","#3A84E6","#997273","#787878","#DB4C6C","#9E7A7A","#554236","#AF5F3C","#93796C","#F9BD3F","#DAB370","#877F6C","#268785"]
    colors_to_plt = [plot_color[i%len(plot_color)] for i in df_merged.iloc[:,-3].values]

    plt.figure(figsize=(5,5))
    plt.axis('off')
    plt.scatter(df_merged.iloc[:,-1],10000-df_merged.iloc[:,-2],c=colors_to_plt,s=10)
    plt.savefig(final_output_img,dpi=1200,bbox_inches='tight',pad_inches=0)

    if not os.path.isfile(manual_annotation_file):
        print("Manual annnotaion doesn't exist, can't calculate ari.!!!!")
        exit()

    df_man = pd.read_csv(manual_annotation_file,index_col=0)
    df_man.sort_index(inplace=True)

    def calc_ari(df_1, df_2):
        df_merged = pd.merge(df_1, df_2, left_index=True, right_index=True).dropna()
        cols = df_merged.columns
        for col in cols:
            df_merged[col] = df_merged[col].values.astype('int')
        return adjusted_rand_score(df_merged[cols[0]].values, df_merged[cols[1]].values)

    ari = calc_ari(df_man,df_final)
    if sample == 'Melanoma':
        df_manual_partial = pd.read_csv(f"{preprocessed_dataset_folder}/{dataset}/{sample}/manual_annotations_wo_unannotated_reg.csv", index_col=0)
        ari = calc_ari(df_manual_partial, df_final.loc[df_manual_partial.index])

    print(f"Ari for dataset:{dataset} sample:{sample} scheme:{scheme} is: ",ari)
    pd.DataFrame([{"ARI":ari}]).to_csv(final_output_ari)

    
