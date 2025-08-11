# SMLM-Analysis
Python Script collection to analyse Single molecule localisation microscopy data collected from dSTROM experiments


## Single molecule localisation scripts
This concerns all scripts and files which are in the SMLM folder.

**File: abberations.mat**
This file contains the aberrations which were calculated using the Maltab Script from 
Alexander Jesacher. 

**Script: automatic analysis different n.ipynb**
Loads experimental dSTORM data (blinking signals) and fits the modelled PSF.
Furthermore, the script calculates the localisations using different defocus values and 
different refractive indices for the middle layer 
The output contains the localisations which are then further analysed using the bead and drift.ipynb 

**Script: bead and drift.ipynb**
Loads the previously calculated localisations together with the recorded bead data and 
corrects for drift in x/y directions . 
The output contains the drift corrected localisations which are then further analysed. 

**Script: Test_light_sheet analysis.ipynb**
File downloaded from Julian Maloberti and adjusted to our needs. 
Script does the same actions as the two scripts “automatic analysis different n” and “bead and drift.ipynb”, but only for one refractive index. Script was used in the analysis to optimise the pre-localisation parameters. Loads the experimental dSTORM data (blinking signals) and fits the modelled PSF. Next the bead drift correction is performed and the localisations are exported for further analysis. 



## Analysis scripts

**Script: 3 loc filtering find defocus.ipynb**
Script for determining the true defocus of the fibril and filtering out bad localisations. 
Additionally, it performs a tilt correction to account for non-horizontal coverslip surfaces. 

**Script: 4 overlay.ipynb**
Script to overlay the SMLM data with the AFM data by a simple transformation using rotation and translation. This ensures that the one and the same fibril is analysed recorded by SMLM and AFM. 

**Script: 5 fitting and points for one fibril.ipynb**
Script to account for the curvature of the collagen fibril for SMLM and AFM data.
Output of the localisations is in the transformed coordinations.

**Script: 6 results analysis.ipynb**
Script to analyse the collagen cross sections. Localisations are filtered and the heights of the cross sections are determined using a sliding window. This process is repeated for the different refractive indices and the true refractive index of the sample is determined using the AFM height and the SMLM results. Finally an error estimation is performed, using the resampling method.

**Script: 7 comparing_results.ipynb**
Script that analyses and displays the results which have been determined in step 6. 

**Script: functions.ipynb**
Collection of functions that have been used in the scripts.
