#! /bin/bash
config_file_expert="configs/melanoma/melanoma_config_expert.json"
config_file_mclust="configs/melanoma/melanoma_config_mclust.json"

# Rscript get_genex_data_from_rds_ST_data.R ${config_file_expert}

python melanoma_to_matrix_representation_converter.py --params ${config_file_mclust}
echo "========================================"
echo "Data convertend in matrix representation"
python autoscribble_dom.py --params ${config_file_mclust}
echo "========================================"
echo "Model run complete"
python best_model_estimator.py --params ${config_file_mclust}
echo "========================================"
echo "Best model evaluated with goodness score"
python show_results.py --params ${config_file_mclust}

# python melanoma_to_matrix_representation_converter.py --params ${config_file_expert}
# echo "========================================"
# echo "Data convertend in matrix representation"
# python scribble_dom.py --params ${config_file_expert}
# echo "========================================"
# echo "Model run complete"
# python best_model_estimator.py --params ${config_file_expert}
# echo "========================================"
# echo "Best model evaluated with goodness score"
# python show_results.py --params ${config_file_expert}