#!/bin/bash

# Define the threshold and the input file
threshold=2.3
input_file="PCC_group.gfeat/cope1.feat/thresh_zstat1.nii.gz"
standard_space_img="$FSLDIR/data/standard/MNI152_T1_2mm_brain"

# Base input filename without extension
base_input_file=$(basename $input_file .nii.gz)

# Ensure the input file exists
if [ ! -f "$input_file" ]; then
    echo "Input file does not exist: $input_file"
    exit 1
fi

# Binarize the thresholded image to identify significant voxels
binary_img="${base_input_file}_binary"
fslmaths $input_file -thr $threshold -bin ${binary_img}.nii.gz

# Ensure binary image was created
if [ ! -f "${binary_img}.nii.gz" ]; then
    echo "Failed to create binary image: ${binary_img}.nii.gz"
    exit 1
fi

# Use atlasquery to query the atlas for brain regions corresponding to significant voxels
atlas_path="$FSLDIR/data/atlases/HarvardOxford/HarvardOxford-sub-prob-2mm"
output_regions="${base_input_file}_regions.txt"

# Using atlasquery to find regions from the atlas that correspond to significant activation
atlasquery -i $standard_space_img -a "Harvard-Oxford Cortical Structural Atlas" -m ${binary_img}.nii.gz -o $output_regions

# Output the results
echo "Brain regions corresponding to significant voxels:"
cat $output_regions