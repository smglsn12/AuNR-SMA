# AuNR-SMA
The code base accompanying the manuscript "Automated Gold Nanorod Spectral Morphology Analysis Pipeline" (https://pubs.acs.org/doi/10.1021/acsnano.4c09753)

Python files: 
"AuNR_Automated_Analysis.py" contains an object, called Absorption_decon, which runs most of the code discussed in this work. A few additional visualization and computation functions are included in this script as well.
"Input_Object_Creation.py" contains a series of objects and functions that handles bulk visualization of the model validation samples used in this work. 

Notebooks:
"AuNR_SMA_Use_and_analysis.ipynb" contains code showing how the model was used in Applications 1 and 3 (Figure 3 - discussed in the manuscript) and how the model was validation against labeled spectra (Figure 2). 
It also generates a few example plots used to make the flowchat in Figure 4. 

"Edge_cutoff_threshold_selection.ipynb" creates the plot and illustrates the rational used to determine the edge cutoff threshold (795 nm) shown in Figure 4. 

"Figure 1 and overlap examples.ipynb" generates the plots used in figure 1 and the overlap metric/2d population distribution projection example figures in the SI 

"NaBH4_in_diglyme_experiments.ipynb" and "NaBH4_in_diglyme_experiments_2.ipynb" show how the HT runs were used to validate digylme as a viable solvent for NaBH4 for HT synthesis, as discussed in the methods section and SI 

"flowchart_overlap_bar_plots.ipynb" contains the bar plots shown in the flow chart in figure 4 justifying the different fitting procedures used by the model, depending on spectral shape, and the overall procedure throwing out uncertian fits

"high_throughput_validation_data_processing.ipynb" and "raw_data_analysis_unlabeled_HT_run.ipynb" show how the high throughput data was processed and analyzed 

The project data files can be found at https://drive.google.com/drive/folders/10ib3a317GnvsV7skDOs4LRmjGoQxSwdk?usp=sharing. The folders names are linked to data loading calls found in the notebooks/python files. 
The data in "Au NR Profiles (for size prediction model)" is needed to use the objects in "AuNR_Automated_Analysis.py" and much of the remaining data and folders are used in AuNR_SMA_Use_and_analysis.ipynb. Increased organization of this dataset and a dryad publication is in progress, check back here for updates! 

Relevant versions: Numpy Version: 1.21.5
Matplotlib Version: 3.5.1
Pandas Version: 1.3.5
Python Version: 3.7.11
Numba Version: 0.53.1
Joblib Version: 1.1.0
