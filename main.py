import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
import ecg_reader
import numpy as np
import warnings
import os

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


if __name__ == "__main__":

    # note - this code does require GPU access to run

    print('Use the ECG VAE for identification of high-risk features in HCM')

    # Load a sample ECG saved in XML format
    # --- note - based on specific XML file formatting, ECG reader xml extraction may need to be locally adjusted
    # view 12 lead ECG with QRS identification
    # view the median beat identification algorithm

    samp_num = 1

    if samp_num == 1:
        age = 38  # in years
        sex = 0  # 0 = female, 1 = male

    ecg = ecg_reader.extract_ecg_xml_muse(f'Sample_ECG\\Sample_ECG_{samp_num}')
    ecg.plot_ecg()
    plt.suptitle(f'12 Lead View - Sample {samp_num}')
    ecg.plot_median_beat(plot_all_beats=0)
    plt.suptitle(f'Median Beat Identification for Sample {samp_num}')
    heart_rate = np.round(ecg.freq/np.mean(np.diff(ecg.qrs_peaks))*60)

    # Load the VAE
    # visualize reconstruction of the sample ECG
    # print the latent variable encoding of the ECG

    model = tf.keras.models.load_model('VAE_HCM_HR_features')

    ecg_reconstruction = ecg_reader.empty_ecg()
    enc_mean, enc_logvar, encoding = model.encoder(ecg.median_beat[tf.newaxis, ...])
    ecg_reconstruction.median_beat = model.decoder(encoding)[0,:,:]

    fig, gs = ecg.plot_median_beat(plot_all_beats=0,color='black',new_fig=1)
    ecg_reconstruction.plot_median_beat(color='r',new_fig=0, old_fig=[fig, gs], alpha=0.7)
    plt.legend(['Original', 'Reconstruction'], loc='lower left')
    plt.suptitle(f'Reconstruction of Sample {samp_num}')

    print(f'\nLatent variables for Sample {samp_num}:')
    print(np.array(encoding)[0])

    # Load classifiers for each of the four high-risk features
    # Systolic Dysfunction (LVEF < 50%)
    # Massive Hypertrophy (Wall Thickness > 3.0cm)
    # LV apical aneurysm
    # Extensive LGE (>15% of LV myocardial volume)

    # print the model predicted presence/absence of high-risk feature
    # print the overall recommendation for CMR based on presence of ANY high-risk feature

    hr_features = ['systolic_dysfunction', 'massive_hypertrophy','apical_aneurysm', 'extensive_lge']
    use_covars = [0,  1, 0, 1,]
    cutoff = [0.039, 0.082, 0.120, 0.100] # cutoffs may need to be calibrated for particular HCM populations
    recommend_cmr = 0
    print()
    for i, hr_feature in enumerate(hr_features):
        print(hr_feature)
        classifier = tf.keras.models.load_model('Classifier_Models\\' + hr_feature)
        if use_covars[i] == 0:
            continuous_prediction = classifier(encoding)[0][0].numpy()
        else:
            covars = [age, sex, heart_rate]
            continuous_prediction = classifier((encoding, np.array(covars)[tf.newaxis, ...]))[0][0].numpy()
        if continuous_prediction > cutoff[i]:
            dichotomous_prediction = 'Present'
            recommend_cmr = 1
        else:
            dichotomous_prediction = 'Absent'
        print(f'  Prediction: ' + dichotomous_prediction)
    print()
    if recommend_cmr:
        print('CMR is recommended for further evaluation of High-risk features')
    else:
        print('No CMR is recommended due to low probability of High-risk features')
    plt.show()

