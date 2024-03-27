import os

# Define the directory to start from
base_directory = '/rds/projects/c/chechlmy-chbh-mricn/axs2210/recon'

# Mapping of old filenames to new filenames
rename_map = {
    'fs004a001.nii.gz': 'fmri1.nii.gz',
    'fs005a001.nii.gz': 'fmri2.nii.gz',
    'fs006a001.nii.gz': 'fmri3.nii.gz',
}

# Set of new filenames to avoid renaming conflicts
new_filenames = set(rename_map.values())

def rename_files(root_directory):
    for dirpath, dirnames, filenames in os.walk(root_directory):
        for filename in filenames:
            # Check if the file is one that we want to rename and it's not already one of the new filenames
            if filename in rename_map and filename not in new_filenames:
                old_path = os.path.join(dirpath, filename)
                new_path = os.path.join(dirpath, rename_map[filename])
                
                # Ensure the new filename does not already exist in the directory
                if not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    print(f"Renamed '{old_path}' to '{new_path}'")
                else:
                    print(f"Skipped '{old_path}' because '{new_path}' already exists")

if __name__ == "__main__":
    rename_files(base_directory)