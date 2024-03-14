## Magnetic Resonance Imaging in Cognitive Neuroscience (MRICN)<br>University of Birmingham (Spring 2024)

These are the materials for my input to the MRICN module at the University of Birmingham. 

### Aamir vs Chris (An introduction to neuroimaging data visualization and analysis using nibabel and nilearn)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sohaamir/MRICN/blob/main/aamir_vs_chris/aamir_vs_chris.ipynb) 
[![Render Notebook](https://img.shields.io/badge/render-nbviewer-orange?logo=jupyter)](https://nbviewer.org/github/sohaamir/MRICN/blob/main/aamir_vs_chris/aamir_vs_chris.ipynb)

The `aamir_vs_chris.ipynb` is a Google Colab notebook demonstrating the basic functions of two Python packages, `nibabel` and `nilearn` for manipulating and plotting neuroimaging data. [Nibabel](https://nipy.org/nibabel/) is a Python library primarily used for reading and writing different types of neuroimaging data files, while [Nilearn](https://nilearn.github.io/stable/index.html) is a Python module used for conducting a wide variety of statistical analyses on neuroimaging data.

It is not necessary as part of the module to know this, and we are only scratching the surface of what `nibabel` and `nilearn` can do, but it presents an alternative approach to the conventional GUI-based alternative of FSL/FSLeyes.

Specifically, we go through the following uses: 

### Nibabel
- Accessing the NIFTI metadata
- Data visulization and comparison
- Creating Animated GIFs from MRI Scans

### Nilearn
- Visualizing Brain Overlays
- Plotting Brain Images with Nilearn
- Statistical Analysis of Brain Images

We then can practically demonstrate a simple case of statistical analysis by comparing the size of two brains, my own and [Chris Gorgolewski's](https://github.com/chrisgorgo), a prime contributor to many open-source initiatives in neuroimaging, who's structural T1 scan is made publicly available through [OpenNeuro](openneuro.org). 

Code for running the voxel comparison test in FSL: 

```bash
# Calculate the number of non-zero voxels for Chris' brain
chris_voxels=$(fslstats chris_bet.nii -V | awk '{print $1}')

# Calculate the number of non-zero voxels for Aamir's brain
aamir_voxels=$(fslstats aamir_bet.nii -V | awk '{print $1}')

# Print the number of voxels
echo "Chris' brain size in voxels: $chris_voxels"
echo "Aamir's brain size in voxels: $aamir_voxels"

# Compare the voxel counts and print who has the bigger brain
if [ "$chris_voxels" -gt "$aamir_voxels" ]; then
    echo "Chris has the bigger brain."
elif [ "$chris_voxels" -lt "$aamir_voxels" ]; then
    echo "Aamir has the bigger brain."
else
    echo "Both brains are the same size."
fi
```
