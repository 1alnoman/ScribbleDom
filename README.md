# ScribbleDom
A method to find spatial domain from Spatial Transcriptomics data using scribble annotated histology image, or using 
output of other possibly non-spatial spatial domain detection method (e.g. mclust).

# Prerequisites
Recommended Python version: 3.10.6</br>
Recommended R version: 4.3.1</br>
Recommended conda version: 4.12.0

To ensure reproducibility, the following is done:
```
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
Note that, you do not need to set these if you run the program in cpu instead of gpu.

# Installation
First set and activate your environment by using the following command:
```
conda env create -f environment.yml
conda activate scribble_dom
```

# Run experiments with availabe data.
To run human breast cancer data :
```
chmod +x run_bcdc_ffpe.sh
./run_bcdc_ffpe.sh
```
To run melanoma cancer data :
```
chmod +x run_melanoma.sh
./run_melanoma.sh
```
To run human dlpfc cancer data :
```
chmod +x run_human_dlpfc.sh
./run_human_dlpfc.sh
```


# To run other visium/st data

## step - 1: 
 Prepear a ```config_mclust.json``` and a ```config_expert.json``` file. You will get an example in ```configs/bcdc/bcdc_config_expert.json``` file and prepear raw count matrix data.

```json
{
    "preprocessed_data_folder" : "preprocessed_data",
    "matrix_represenation_of_ST_data_folder" : "matrix_representation_of_st_data",
    "model_output_folder" : "model_outputs",
    "final_output_folder" : "final_outputs",

    "space_ranger_output_directory" : "raw_gene_x",
    "dataset": "cancers",
    "samples": ["bcdc_ffpe"],
    "technology": "visium",
    "n_pcs": 15,
    "n_cluster_for_auto_scribble": 2,
    "schema": "mclust",

    "max_iter": 300,
    "nConv": 1,
    "seed_options": [4],
    "alpha_options": [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],
    "beta_options": [0.25,0.3,0.35,0.4],
    "lr_options": [0.1]
}
```
1. "dataset" : Yous should give a name of your dataset for example here the name is "cancers". This is for internal file structure in our system.
2. "samples" : You should give a name of the sample in your dataset. Here the sample is bcdc_ffpe. This is also for internal file structure in our system.
3. "technology" : This can be visium/st for this pipeline.
4. "n_pcs" : The number of principal components you want for your data.
5. "n_cluster_for_auto_scribble" : This field is used for automatic scribble generation. Give number of cluster for mclust initialization of your data.
6. "schema": This can be either expert/mclust to indicate use of expert generated scribble or automated scribble.
7. "space_ranger_output_directory" : This field should contains the space ranger output for mat directory for count matrix data.
This directory structure is (for visum data):
```
.
└── {space_ranger_output_directory}/
    └── {dataset}/
        └── {samples[i]}/
            ├── spatial/
            │   ├── tissue_positions_list.csv
            │   ├── scalefactors_json.json
            │   ├── tissue_hires_image.png
            │   └── tissue_lowres_image.png
            └── {samples[i]}_filtered_feature_bc_matrix.h5
```
or (for st data):
```
.
└── {space_ranger_output_directory}/
    └── {dataset}/
        └── {samples[i]}/
            └── {samples[i]}.rds
```
For example, for the json file shown above, the structure of space_ranger_output_directory should be:
```
.
└── raw_gene_x/
    └── cancers/
        └── bcdc_ffpe/
            ├── spatial/
            │   ├── tissue_positions_list.csv
            │   ├── scalefactors_json.json
            │   ├── tissue_hires_image.png
            │   └── tissue_lowres_image.png
            └── bcdc_ffpe_filtered_feature_bc_matrix.h5
```

other fileds can be as it is, for your config file.


## step - 2:

Preprocess the data. Finds the top 2000 highly variable genes and calculates principal components from raw Gene expression data. Also runs ```Mclust``` algorithm to create a automated scribble.

For visium data:
```
Rscript get_genex_data_from_10x_h5.R config_mclust.json
```
For st data:
```
Rscript get_genex_data_from_rds_ST_data.R config_mclust.json
```
This step will produce preprocessed data with principal components of spots, coordinates and mclust_backbone.csv as Automated Scribble in the folder below: 
```
.
└── {preprocessed_data_folder}/
    └── {dataset}/
        └── {samples[i]}/
```

## step - 3:
Create a manual scribble (``manual_scribble.csv``) using [Loupe browser](https://support.10xgenomics.com/single-cell-gene-expression/software/visualization/latest/what-is-loupe-cell-browser) and a ```.cloupe``` file from [space ranger output](https://support.10xgenomics.com/spatial-gene-expression/software/pipelines/latest/output/aggr-overview) of your sample. You will get a video [tutorial](https://youtu.be/nRy9TszaduQ) here. Place the manual scribble (manual_scribble.csv) in the folder location: 
```
.
└── {preprocessed_data_folder}/
    └── {dataset}/
        └── {samples[i]}/
            └── manual_scribble.csv

```
If you don't generate ```manual_scribble.csv``` in this step, you will get the result from AutoScribbleDom only.

## step - 4:
To run our pipeline run the commands below:-
For visium data:
```
chmod +x run_other_visium.sh
./run_other_visium.sh config_expert.json config_mclust.json
```
For st data:
```
chmod +x run_other_st.sh
./run_other_st.sh config_expert.json config_mclust.json
```

here config_expert.json and config_mclust.json are same as config.json file described above, with only difference of the scheme field, i.e.
```json
.
.
"schema": "expert",
.
.
```
in config_expert.json file. And


```json
.
.
"schema": "mclust",
.
.
```
in config_mclust.json file.


## step - 5:
Get the final output in final_outputs folder in
```
.
└── final_outputs/
    └── dataset/
        └── samples[i]/
```

# Miscellaneous
## Manual annotation for melanoma st sample
We have generated a manual annotation for melanoma cancer dataset by Thrane et al. We used [ST Spot Detector](https://github.com/SpatialTranscriptomicsResearch/st_spot_detector) to map the spots to high resulation image's pixels. You will get the the csv file mapping the spot to the type of spot [here](preprocessed_data/cancers/Melanoma/manual_annotations.csv). In this ```.csv``` file the mapping corresponds to :</br></br>
'0' : 'Stroma'</br>
'1' : 'Lymphoid tissue'</br>
'2' : 'Melanoma'</br>
'3' : 'Unannotated by publisher'</br>
None : 'Unannotated for being in borders of 2 type'</br>

![A mushroom-head robot](preprocessed_data/cancers/Melanoma/melanoma_manual_annotation.png 'Melanoma Manual Annotation')

## Scribbles for other samples
Get the scribbles for other sample in the directory structure mentioned below:
```
.
└── {preprocessed_data_folder}/
    └── {dataset}/
        └── {samples[i]}/
            └── manual_scribble.csv
            └── manual_annotation.csv

```

# Licenses and Cites

License: [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)</br>
Cite the code on Zenodo: [![DOI](https://zenodo.org/badge/681572669.svg)](https://zenodo.org/badge/latestdoi/681572669)</br>
Cite as:
```
ScribbleDom: Using Scribble-Annotated Histology Images to Identify Domains in Spatial Transcriptomics Data
Rahman, Mohammad Nuwaisir, Al Noman, Abdullah, Turza, Abir Mohammad, Abrar, Mohammed Abid, Samee, Md. Abul Hassan, and Rahman, M Saifur
```
