# Magnetic Resonance Imaging in Cognitive Neuroscience (MRICN), Spring term, University of Birmingham, 2024

These are the materials for my input to the MRICN module at the University of Birmingham. 

## Aamir vs Chris (An introduction to nibabel and nilearn)

The `aamir_vs_chris.ipynb` is a Google Colab notebook demonstrating the basic functions of two Python packages, `nibabel` and `nilearn` for manipulating and plotting neuroimaging data. [Nibabel](https://nipy.org/nibabel/) is a Python library primarily used for reading and writing different types of neuroimaging data files, while [Nilearn](https://nilearn.github.io/stable/index.html) is a Python module used for conducting a wide variety of statistical analyses on neuroimaging data.

It is not necessary as part of the module to know this, and we are only scratching the surface of what `nibabel` and `nilearn` can do, but it presents an alternative approach to the conventional GUI-based alternative of FSL/FSLeyes.

Specifically, we go through the following uses: 

### Nibabel
- Accessing the NIFTI metadata
- Data visulization and comparison
- Creating Animated GIFs from MRI Scans

### Nilearn
- Visualizing Brain Overlays
- Extracting Connectome Features
- Performing Statistical Analysis on Brain Images
- Plotting Brain Images with Nilearn
- Statistical Analysis of Brain Images

We then can practically demonstrate a simple case of statistical analysis by comparing the size of two brains, my own and [Chris Gorgolewski's](https://github.com/chrisgorgo), a prime contributor to many open-source initiatives in neuroimaging, who's structural T1 scan is made publicly available through [OpenNeuro](openneuro.org). 
