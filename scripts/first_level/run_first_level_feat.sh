#!/bin/bash

# Define an array with the .fsf file names
fsf_files=(
    p01_s1.fsf p02_s1.fsf p03_s1.fsf p03_s2.fsf p04_s1.fsf
    p04_s2.fsf p05_s1.fsf p05_s2.fsf p05_s3.fsf p06_s1.fsf
    p06_s2.fsf p07_s1.fsf p07_s2.fsf p08_s1.fsf p08_s2.fsf
    p09_s1.fsf p09_s2.fsf p10_s1.fsf p10_s2.fsf p11_s1.fsf
    p11_s2.fsf p12_s1.fsf p12_s2.fsf p13_s1.fsf p13_s2.fsf
    p14_s1.fsf p14_s2.fsf p15_s1.fsf p15_s2.fsf
)

# Loop through the .fsf files and run FEAT on each one
for fsf_file in "${fsf_files[@]}"; do
    # Extract participant ID and session from the file name
    participant_id=$(echo "$fsf_file" | cut -d'_' -f1)
    session=$(echo "$fsf_file" | cut -d'_' -f2 | cut -d'.' -f1)

    echo "Processing participant: $participant_id, Session: $session"

    # Run FEAT analysis
    feat "$fsf_file"

    echo "Completed processing of $participant_id, Session: $session"
done

echo "All FEAT analyses are completed."

