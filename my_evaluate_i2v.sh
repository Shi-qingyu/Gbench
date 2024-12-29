#!/bin/bash

# Define the model list
models=("cogvideox-5b-i2v")

# Define the dimension list
dimensions=("subject_consistency" "overall_consistency" "motion_smoothness" "dynamic_degree")

# Corresponding folder names
category="Vehicle"
folders=(${category} ${category} ${category} ${category})

# Base path for videos
base_path='./sampled_videos/' # TODO: change to local path

# Loop over each model
for model in "${models[@]}"; do
    # Loop over each dimension
    for i in "${!dimensions[@]}"; do
        # Get the dimension and corresponding folder
        dimension=${dimensions[i]}
        folder=${folders[i]}

        # Construct the video path
        videos_path="${base_path}${model}/${folder}"
        echo "$dimension $videos_path"

        # Run the evaluation script
        python -W ignore evaluate.py --full_json_dir "I2V/${folder}/full_info.json" --videos_path $videos_path --dimension $dimension --output_path "./evaluation_results/${model}/${folders[i]}"
    done
done
