#!/bin/bash

# Define the model list
models=("videocrafterv2" "zeroscope" "lavie")

# Define the dimension list
dimensions=("subject_consistency" "background_consistency" "aesthetic_quality" "imaging_quality" "object_class" "multiple_objects" "color" "spatial_relationship" "scene" "temporal_style" "overall_consistency" "human_action" "temporal_flickering" "motion_smoothness" "dynamic_degree" "appearance_style")

# Corresponding folder names
folders=("subject_consistency" "background_consistency" "overall_consistency" "overall_consistency" "object_class" "multiple_objects" "color" "spatial_relationship" "scene" "temporal_style" "overall_consistency" "human_action" "temporal_flickering" "subject_consistency" "subject_consistency" "appearance_style")

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
        python -W ignore evaluate.py --full_json_dir "./full_info.json" --videos_path $videos_path --dimension $dimension --output_path "./evaluation_results/${model}"
    done
done
