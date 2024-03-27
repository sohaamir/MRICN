#!/bin/bash

# Function to create the design.fsf file for the participants
create_design_fsf() {
    participant=$1
    output_dir="/rds/projects/c/chechlmy-chbh-mricn/axs2210/feat/2/${participant}.gfeat"

    # Set FEAT directory paths
    feat_files[1]="/rds/projects/c/chechlmy-chbh-mricn/axs2210/feat/1/${participant}_s1.feat"
    feat_files[2]="/rds/projects/c/chechlmy-chbh-mricn/axs2210/feat/1/${participant}_s2.feat"

    # Create the design.fsf file
    cp /rds/projects/c/chechlmy-chbh-mricn/axs2210/feat/2/p03.gfeat/design.fsf "${participant}_design.fsf"
    sed -i "s|set fmri(outputdir) .*|set fmri(outputdir) \"$output_dir\"|" "${participant}_design.fsf"
    sed -i "/^set feat_files(1)/d" "${participant}_design.fsf"
    sed -i "/^set feat_files(2)/d" "${participant}_design.fsf"
    sed -i "1aset feat_files(1) ${feat_files[1]}" "${participant}_design.fsf"
    sed -i "2aset feat_files(2) ${feat_files[2]}" "${participant}_design.fsf"
    sed -i "s|set fmri(evtitle1) .*|set fmri(evtitle1) \"$participant\"|" "${participant}_design.fsf"
}

# Now we can create the design.fsf files (excluding the first three)
for participant in p04 p05 p06 p07 p08 p09 p10 p11 p12 p13 p14 p15; do
    echo "Creating design.fsf for $participant..."
    create_design_fsf "$participant"
done

# Run the higher-level analysis for these participants
for fsf_file in *_design.fsf; do
    participant=${fsf_file%_design.fsf}
    if [[ "$participant" != "p01" && "$participant" != "p02" && "$participant" != "p03" ]]; then
        echo "Running higher-level analysis for $participant..."
        feat "$fsf_file"
    fi
done
