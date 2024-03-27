#!/bin/bash

# Prompt the user for the University account name
read -p "Please enter your University account name: " account_name

# Define the base directory with the user-provided account name
base_dir="/rds/projects/c/chechlmy-chbh-mricn/${account_name}/SBC"

echo "Using base directory: $base_dir"

# Loop over each subject's data
for sub in sub2 sub3; do
    # Define the input .nii.gz file for the subject
    input_file="${base_dir}/${sub}/${sub}.nii.gz"
    
    # Define the output FEAT directory
    output_dir="${base_dir}/${sub}.feat"
    
    # Define the custom EV file for the subject
    custom_ev_file="${base_dir}/${sub}/${sub}_PCC.txt"
    
    # Define the .fsf file for the subject
    design_file="${base_dir}/${sub}.fsf"
    
    # Copy the template design file from sub1 and modify it for the current subject
    cp "${base_dir}/sub1.feat/design.fsf" "$design_file"
    
    # Replace the input file path in the design file
    sed -i "s|set feat_files(1) \".*\"|set feat_files(1) \"${input_file}\"|g" "$design_file"
    
    # Replace the output FEAT directory in the design file
    sed -i "s|set fmri(outputdir) \".*\"|set fmri(outputdir) \"${output_dir}\"|g" "$design_file"
    
    # Replace the custom EV file in the design file
    sed -i "s|set fmri(custom1) \".*\"|set fmri(custom1) \"${custom_ev_file}\"|g" "$design_file"

    # Run FEAT analysis
    feat "$design_file"

    # Remove the .fsf file from the SBC directory after running FEAT
    rm -f "$design_file"
done

echo "FEAT analysis completed for sub2 and sub3."
