Please be aware that, in ALL steps in this Github, users must change the directories included, as well as install the corresponding libraries for the  files to run correctly.

The structure of this github is organised as follows:

###

# Result 1

## Step 1: Reading files

Input: Flow cytometry data files obtained in http://flowrepository.org/id/FR-FCM-Z68U, as a Folder, distinguished in two subfolders "Relapsed" (R) and "NonRelapsed" (NR) patients

Output: .txt files, distinguished in two subfolders for R and NR patients, having deleted duplicates in data, with flow cytometry information for each parameter. These two processes are distinguished so that we can keep track of duplicated events in files.

## Step 2: Obtaining landmarks

Input: output from Step 1.

Output: .csv files with the corresponding Landmarks, having performed the MaxMin algorithm.

## Step 3: Vietoris-rips in pairwise combinations

Input: output from Step 2.

Output: .txt files, where for each pairwise combination of parameters, the vietoris-rips method is applied in each patient. Such files include the results of the persistence barcodes in dimensions 0 and 1, and is organised in R and NR folders. Please see that these files are included in the Folder "RIPS DATA (from Step 3-4 and 6-7)".

## Step 4: Statistical analysis

Input: output from Step 3.

Output: .csv files, including for each pairwise combination of parameters, the maximal, minimum, median, mean, standard deviation and length of the persistence barcodes used.

## Step 5: Random Forest

Input: output from Step 4.

Output: Data S1 and S2 from the main manuscript, where AUC and other classification markers are obtained using Random Forest.

###

# Result 2

## Step 6: Vietoris-Rips for Biomarkers CD10-20-38-45

Input: output from Step 2.

Output: Output: .txt files, where for each pairwise combination of parameters of the Biomarkers CD10-20-38-45, the vietoris-rips method is applied in each patient. Such files include the results of the persistence barcodes in dimensions 0 and 1, and is organised in R and NR folders. Please see that these files are included in the Folder "RIPS DATA (from Step 3-4 and 6-7)", in the subfolder "Pairwise combinations".

## Step 7: Statistical analysis

Input: output from Step 6.

Output: persistence curves and their statistical analysis presented in Supplementary Information.

###

# Result 3

## Step 8: Creation of Persistence Images (PIs)

Input: Output from Step 6, where Vietoris Rips was performed only for the 4D data including markers CD10-CD20-CD38-CD45. These data are saved in the Folder "RIPS DATA (from Step 3-4 and 6-7)", in the subfolder 4D Analysis. 

Output: Persistence Images in dimensions 0, 1 and 2. These data are saved in the Folder "RIPS DATA (from Step 3-4 and 6-7)", in the subfolder 4D Analysis. 

## Step 9: First steps in classification with PIs

Input: Output from Step 8. 

Output: Obtention of Discriminant Areas in PIs with Logistic Regression classification and initial classification of PIs with Support Vector Machine

## Step 10: SVM and LR for classification

Input: Output from Step 9.

Output: Classification tables using Logistic Regression and Support Vector Machine for all PIs in each dimension

## Step 11: SVM with/without upper sampling and all PIs together

Input: Output from Step 9.

Output: Classification tables using Support Vector Machine with/without upper sampling and all PIs  in each dimension glued together

###

Please be careful as some of these steps involve having already followed the prior steps.

email: salvador.chulian@uca.es
