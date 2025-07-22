import os, sys, glob, boto3, shutil
from tqdm import tqdm
import numpy as np
import scipy as sp
import pandas as pd
import scipy.io as sio
from nilearn import datasets
import nibabel as nib

# setup
s3 = boto3.resource('s3',
                    aws_access_key_id='',
                    aws_secret_access_key='',
                    )
bucket = s3.Bucket('hcp-openaccess')

# proj_dir = '/projectsp/f_lp756_1/lindenmp/research_data/HCP_YA'
proj_dir = '/scratch/f_ah1491_1/open_data/HCP_YA'
# proj_dir = '/mnt/storage_ssd_raid/research_data/HCP_YA'
subject_ids = np.loadtxt(os.path.join(proj_dir, 'HCPYA_Schaefer4007_subjids.txt'), dtype=str)
# subject_ids = subject_ids[:2]
n_subs = len(subject_ids)
print(n_subs)
print(subject_ids)

# output direcotry
s1200_dir = os.path.join(proj_dir, 'HCP_1200')
print(s1200_dir)
if not os.path.exists(s1200_dir):
    os.makedirs(s1200_dir)
    
# download data
files_to_download = [
    'HCP_1200/subject_id/T1w/Diffusion/bvals',
    'HCP_1200/subject_id/T1w/Diffusion/bvecs',
    'HCP_1200/subject_id/T1w/Diffusion/data.nii.gz',
    'HCP_1200/subject_id/T1w/Diffusion/nodif_brain_mask.nii.gz',
    'HCP_1200/subject_id/T1w/T1w_acpc_dc_restore_brain.nii.gz',
    'HCP_1200/subject_id/MNINonLinear/fsaverage_LR32k/subject_id.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii',
    'HCP_1200/subject_id/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii',
    'HCP_1200/subject_id/MNINonLinear/Results/rfMRI_REST2_LR/rfMRI_REST2_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii',
    'HCP_1200/subject_id/MNINonLinear/Results/rfMRI_REST1_RL/rfMRI_REST1_RL_Atlas_MSMAll_hp2000_clean.dtseries.nii',
    'HCP_1200/subject_id/MNINonLinear/Results/rfMRI_REST2_RL/rfMRI_REST2_RL_Atlas_MSMAll_hp2000_clean.dtseries.nii',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_EMOTION_LR/tfMRI_EMOTION_LR_Atlas_MSMAll.dtseries.nii',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_GAMBLING_LR/tfMRI_GAMBLING_LR_Atlas_MSMAll.dtseries.nii',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_LANGUAGE_LR/tfMRI_LANGUAGE_LR_Atlas_MSMAll.dtseries.nii',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_MOTOR_LR/tfMRI_MOTOR_LR_Atlas_MSMAll.dtseries.nii',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_RELATIONAL_LR/tfMRI_RELATIONAL_LR_Atlas_MSMAll.dtseries.nii',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_SOCIAL_LR/tfMRI_SOCIAL_LR_Atlas_MSMAll.dtseries.nii',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_WM_LR/tfMRI_WM_LR_Atlas_MSMAll.dtseries.nii',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_EMOTION_RL/tfMRI_EMOTION_RL_Atlas_MSMAll.dtseries.nii',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_GAMBLING_RL/tfMRI_GAMBLING_RL_Atlas_MSMAll.dtseries.nii',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_LANGUAGE_RL/tfMRI_LANGUAGE_RL_Atlas_MSMAll.dtseries.nii',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_MOTOR_RL/tfMRI_MOTOR_RL_Atlas_MSMAll.dtseries.nii',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_RELATIONAL_RL/tfMRI_RELATIONAL_RL_Atlas_MSMAll.dtseries.nii',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_SOCIAL_RL/tfMRI_SOCIAL_RL_Atlas_MSMAll.dtseries.nii',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_WM_RL/tfMRI_WM_RL_Atlas_MSMAll.dtseries.nii',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_EMOTION/tfMRI_EMOTION_hp200_s2_level2_MSMAll.feat/subject_id_tfMRI_EMOTION_level2_hp200_s2_MSMAll.dscalar.nii',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_EMOTION/tfMRI_EMOTION_hp200_s2_level2_MSMAll.feat/Contrasts.txt',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_GAMBLING/tfMRI_GAMBLING_hp200_s2_level2_MSMAll.feat/subject_id_tfMRI_GAMBLING_level2_hp200_s2_MSMAll.dscalar.nii',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_GAMBLING/tfMRI_GAMBLING_hp200_s2_level2_MSMAll.feat/Contrasts.txt',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_LANGUAGE/tfMRI_LANGUAGE_hp200_s2_level2_MSMAll.feat/subject_id_tfMRI_LANGUAGE_level2_hp200_s2_MSMAll.dscalar.nii',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_LANGUAGE/tfMRI_LANGUAGE_hp200_s2_level2_MSMAll.feat/Contrasts.txt',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_MOTOR/tfMRI_MOTOR_hp200_s2_level2_MSMAll.feat/subject_id_tfMRI_MOTOR_level2_hp200_s2_MSMAll.dscalar.nii',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_MOTOR/tfMRI_MOTOR_hp200_s2_level2_MSMAll.feat/Contrasts.txt',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_RELATIONAL/tfMRI_RELATIONAL_hp200_s2_level2_MSMAll.feat/subject_id_tfMRI_RELATIONAL_level2_hp200_s2_MSMAll.dscalar.nii',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_RELATIONAL/tfMRI_RELATIONAL_hp200_s2_level2_MSMAll.feat/Contrasts.txt',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_SOCIAL/tfMRI_SOCIAL_hp200_s2_level2_MSMAll.feat/subject_id_tfMRI_SOCIAL_level2_hp200_s2_MSMAll.dscalar.nii',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_SOCIAL/tfMRI_SOCIAL_hp200_s2_level2_MSMAll.feat/Contrasts.txt',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_WM/tfMRI_WM_hp200_s2_level2_MSMAll.feat/subject_id_tfMRI_WM_level2_hp200_s2_MSMAll.dscalar.nii',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_WM/tfMRI_WM_hp200_s2_level2_MSMAll.feat/Contrasts.txt',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_EMOTION_LR/EVs/fear.txt',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_EMOTION_LR/EVs/neut.txt',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_EMOTION_RL/EVs/fear.txt',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_EMOTION_RL/EVs/neut.txt',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_GAMBLING_LR/EVs/win.txt',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_GAMBLING_LR/EVs/loss.txt',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_GAMBLING_RL/EVs/win.txt',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_GAMBLING_RL/EVs/loss.txt',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_LANGUAGE_LR/EVs/story.txt',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_LANGUAGE_LR/EVs/math.txt',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_LANGUAGE_RL/EVs/story.txt',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_LANGUAGE_RL/EVs/math.txt',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_RELATIONAL_LR/EVs/relation.txt',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_RELATIONAL_LR/EVs/match.txt',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_RELATIONAL_RL/EVs/relation.txt',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_RELATIONAL_RL/EVs/match.txt',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_SOCIAL_LR/EVs/mental.txt',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_SOCIAL_LR/EVs/rnd.txt',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_SOCIAL_RL/EVs/mental.txt',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_SOCIAL_RL/EVs/rnd.txt',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_WM_LR/EVs/0bk_body.txt',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_WM_LR/EVs/0bk_faces.txt',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_WM_LR/EVs/0bk_places.txt',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_WM_LR/EVs/0bk_tools.txt',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_WM_LR/EVs/2bk_body.txt',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_WM_LR/EVs/2bk_faces.txt',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_WM_LR/EVs/2bk_places.txt',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_WM_LR/EVs/2bk_tools.txt',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_WM_RL/EVs/0bk_body.txt',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_WM_RL/EVs/0bk_faces.txt',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_WM_RL/EVs/0bk_places.txt',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_WM_RL/EVs/0bk_tools.txt',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_WM_RL/EVs/2bk_body.txt',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_WM_RL/EVs/2bk_faces.txt',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_WM_RL/EVs/2bk_places.txt',
    'HCP_1200/subject_id/MNINonLinear/Results/tfMRI_WM_RL/EVs/2bk_tools.txt'
]

subj_filt = np.zeros(n_subs).astype(bool)

for s, subject_id in enumerate(subject_ids):
    print(s, subject_id)
    
    # freesurfer data
    for s3_object in bucket.objects.filter(Prefix='HCP_1200/{0}/T1w/{0}'.format(subject_id)):
        remote_file = s3_object.key
        remote_path, file = os.path.split(remote_file)
        local_path = os.path.join(proj_dir, remote_path)
        local_file = os.path.join(local_path, file)

        if os.path.isfile(local_file):
            # print('File found!')
            pass
        else:
            print('Local file NOT found!')
            try:
                if not os.path.exists(local_path):
                    os.makedirs(local_path)
                    
                print('\tDownloading: {0}-->{1}'.format(file, local_file))
                bucket.download_file(remote_file, local_file)
            except:
                print('\t\tDownload failed!')
                shutil.rmtree(local_path)
                subj_filt[s] = True

    # diffusion data (& select T1w data)
    remote_files = [sub.replace('subject_id', subject_id) for sub in files_to_download]

    for remote_file in remote_files:
        local_file = os.path.join(proj_dir, remote_file)
        local_path, file = os.path.split(local_file)
        # print(local_path)

        if os.path.isfile(local_file):
            # print('File found!')
            pass
        else:
            print('Local file NOT found!')
            try:
                if not os.path.exists(local_path):
                    os.makedirs(local_path)
                    
                print('\tDownloading: {0}-->{1}'.format(file, local_file))
                bucket.download_file(remote_file, local_file)
            except:
                print('\t\tDownload failed!')
                shutil.rmtree(os.path.join(local_path))
                subj_filt[s] = True

# subject_ids = subject_ids[~subj_filt]
# np.savetxt(os.path.join(proj_dir, 'subject_ids_filtered.txt'), subject_ids, fmt='%s')
# np.savetxt(os.path.join(proj_dir, 'subject_ids_filtered_task.txt'), subject_ids, fmt='%s')
