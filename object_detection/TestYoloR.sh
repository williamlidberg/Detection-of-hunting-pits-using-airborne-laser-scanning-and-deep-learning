#!/bin/bash

# List with subfolders for indices
subdirs=("depthinsink" "elevation_above_pit" "hillshade" "maxelevationdeviation" "maximal_curvature" "minimal_curvature"
                 "multiscaleelevationpercentile" "multiscale_stdon" "profile_curvature" "stdon")

# Decide witch scale to run the evaluation on. 05m or 1m
scale="1m"

# Base-path
base_path="datasets/final_data_${scale}_normalized/testing"

# Iterate over all folder in the list above.
for subdir in "${subdirs[@]}"; do
    full_path="${base_path}/${subdir}"
    images_path="${base_path}/images"
    echo "--- Running experiments for ${subdir} ---"

    # Rename
    echo "Renaming ${full_path} to -----> ${images_path}"
    mv "${full_path}" "${images_path}"

    # Run tests
    python TestYolor.py --data datasets/final_data_${scale}_normalized/testing/data.yaml --img 256 --batch 32 --conf 0.001 --iou 0.5 --device cpu --cfg configurations/yolor_configurations/1_class/yolor_p6_fangstgropar_256x256.cfg --weights eval/models/${scale}/${subdir}/best_overall.pt --names configurations/fangstgropar.names --save-txt --save-conf --verbose --name ${subdir} --mlflow --mlflow_path /mnt/mlflow_tracking/mlruns --experiment_name TrappingPits_${scale}_Testing --run_name ${subdir}

    # Restore
    echo "Restoring ${images_path} to -----> ${full_path}"
    mv "${images_path}" "${full_path}"
    sleep 2
done
