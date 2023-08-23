#! /bin/bash
config_file_expert=$1
config_file_mclust=$2

# Rscript get_genex_data_from_10x_h5.R ${config_file_expert}

python visium_data_to_matrix_representation_converter.py --params ${config_file_mclust}
echo "========================================"
echo "Data convertend in matrix representation"
python autoscribble_dom.py --params ${config_file_mclust}
echo "========================================"
echo "Model run complete"
python best_model_estimator.py --params ${config_file_mclust}
echo "========================================"
echo "Best model evaluated with goodness score"
# python show_results.py --params ${config_file_mclust}

python visium_data_to_matrix_representation_converter.py --params ${config_file_expert}
echo "========================================"
echo "Data convertend in matrix representation"
python scribble_dom.py --params ${config_file_expert}
echo "========================================"
echo "Model run complete"
python best_model_estimator.py --params ${config_file_expert}
echo "========================================"
echo "Best model evaluated with goodness score"
# python show_results.py --params ${config_file_expert}