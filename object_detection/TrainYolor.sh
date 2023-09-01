#!/bin/bash
# Trapping pit experiments

# List of subdirectories for the indices
subdirs=("depthinsink" "elevation_above_pit" "hillshade" "maxelevationdeviation" "maximal_curvature" "minimal_curvature" 
                 "multiscaleelevationpercentile" "multiscale_stdon" "profile_curvature" "stdon")
# Base path, point this to the 0.5m or 1m traing data folder.
base_path="datasets/final_data_05m_normalized/training"

# Run training for all indices in the "subdirs" list above. This will handle renaming of the folders.
for subdir in "${subdirs[@]}"; do
    full_path="${base_path}/${subdir}"
    images_path="${base_path}/images"
    echo "--- Running experiments for ${subdir} ---"

    # Rename path
    echo "Renaming ${full_path} to -----> ${images_path}"
    mv "${full_path}" "${images_path}"
    
    # Run traing
    # Without using pretrained weights from lunar dataset.
    #python TrainYolor.py --cfg configurations/yolor_configurations/1_class/yolor_p6_fangstgropar_256x256.cfg --data datasets/final_data_05m_normalized/training/data.yaml --epochs 300 --weights '' --device 0 --batch-size 32 --img-size 256 256 --hyp configurations/yolor_configurations/1_class/hyp.fangstgropar.yaml --mlflow --mlflow_path /mnt/mlflow_tracking/mlruns --experiment_name William_YolorR_rnd_weights_1m_big-batch_shrinked_labels_no_pretrain --run_name ${subdir}

    # Run training using pretrained weights from lunar dataset.
    python TrainYolor.py --cfg configurations/yolor_configurations/1_class/yolor_p6_fangstgropar_256x256.cfg --data datasets/final_data_1m_normalized/training/data.yaml --epochs 300 --weights 'cleared_lunar.pt' --device 0 --batch-size 32 --img-size 256 256 --hyp configurations/yolor_configurations/1_class/hyp.fangstgropar.yaml --mlflow --mlflow_path /mnt/mlflow_tracking/mlruns --experiment_name William_YolorR_rnd_weights_1m_big-batch_shrinked_labels_pretrained_2 --run_name ${subdir}

    # Restore folder name
    echo "Restore ${images_path} to -----> ${full_path}"
    mv "${images_path}" "${full_path}" 
    sleep 2
done
