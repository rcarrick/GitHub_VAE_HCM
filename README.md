This is a repository for python-based code used to run an ECG variational autoencoder + dense network classifier for prediction of high-risk imaging features in patients with hypertrophic cardiomyopathy.

Running the main file will excute the following actions:

1) Load and view a sample ECG ".xml" file and view the results in standard 12-lead ECG format
2) Use a Pan-Tompkins based algorithm to identify QRS and extract median beat data for each of the 12 leads, then view median beat data in 12-lead format
3) Load the pre-trained ECG variational autoencoder (VAE), and demonstrate median beat reconstruction of the sample ECG.
    a) note - this relies on tensorflow python deep-learning library and requires access to GPU.
    b) Input to the ECG VAE is 12 lead, 250Hz, 1.2s median-beat voltage wave-form data.
4) Print the latent variable (n=24) encoding of the sample ECG
5) Load dense-network classifier models for each of the 4 high-risk imaging features in HCM:
    a) systolic dysfunction (LVEF < 50%)
    b) massive hypertrophy (LV wall thickness > 3.0cm)
    c) LV apical aneurysm 
    d) Extensive late gadolineum enhancement (LGE > 15% myocardial mass)
7) Generate predictions for presence/absence of each high-risk feature for the sample ECG.
    a) Note - these dichotomous predictions are based on previously determined thresholds that provide 80% sensitivity for feature detection in  the original model derivation cohort
8) Generate a recommendation for CMR imaging based on ECG wave-form data
