#!/bin/bash
#SBATCH --requeue                 # Return job to the queue if preempted
#SBATCH --job-name=parcdat        # Assign a short name to your job
#SBATCH --nodes=1                 # Number of nodes you require
#SBATCH --ntasks=1                # Total # of tasks across all nodes
#SBATCH --mem=8000                # Real memory (RAM) required (MB)
#SBATCH --time=01:00:00           # Total run time limit (D-HH:MM:SS)
#SBATCH --output=slurm_%A_%a.out  # STDOUT output file

#SBATCH --partition=main          # Partition (job queue)
# SBATCH --partition=p_dz268_1
#SBATCH --cpus-per-task=1         # Cores per task (>1 if multithread tasks)
# SBATCH --array=1-485
#SBATCH --array=486-969

# get CPU core count
if [[ ${USER} == "lindenmp" ]]; then
    NPROC="$(nproc --all)"
    NPROC=$((NPROC / 4))
elif [[ ${USER} == "lp756" ]]; then
    NPROC=${SLURM_CPUS_PER_TASK}
fi
echo "NRPOC:" ${NPROC}

# Print the task id.
if [[ -z ${SLURM_ARRAY_TASK_ID} ]]; then
    IDX=1
else
    IDX=${SLURM_ARRAY_TASK_ID}
fi
echo "IDX:" ${IDX}

####################################################################################################
# static directories
if [[ ${USER} == "lindenmp" ]]; then
    RESDATA_DIR=/mnt/storage_ssd_raid/research_data
    export FREESURFER_HOME=/usr/local/freesurfer
    TEMPLATEFLOW_HOME=${HOME}/templateflow
    ATLAS_DIR=/home/lindenmp/research_projects/snaplab_tools/data/atlases
elif [[ ${USER} == "lp756" ]]; then
    RESDATA_DIR=/scratch/f_ah1491_1/open_data
    export FREESURFER_HOME=${HOME}/freesurfer
    TEMPLATEFLOW_HOME=${RESDATA_DIR}/templateflow
    ATLAS_DIR=${RESDATA_DIR}/atlases
fi

PROJ_DIR=${RESDATA_DIR}/HCP_YA
S1200_DIR=${PROJ_DIR}/HCP_1200
DERIVS_DIR=${PROJ_DIR}/derivatives
####################################################################################################

####################################################################################################
# subject variables
SUBJECT_ID=$(sed -n ${IDX}p ${PROJ_DIR}/HCPYA_Schaefer4007_subjids.txt)
echo ${SUBJECT_ID}

MYELIN_OUTDIR=${DERIVS_DIR}/myelin_parcellated/${SUBJECT_ID}
rm -rf ${MYELIN_OUTDIR}
mkdir -p ${MYELIN_OUTDIR}

FMRI_OUTDIR=${DERIVS_DIR}/fmri_parcellated/${SUBJECT_ID}
rm -rf ${FMRI_OUTDIR}
mkdir -p ${FMRI_OUTDIR}

FMRI_CON_OUTDIR=${DERIVS_DIR}/fmri_parcellated_task-contrasts/${SUBJECT_ID}
rm -rf ${FMRI_CON_OUTDIR}
mkdir -p ${FMRI_CON_OUTDIR}
####################################################################################################

####################################################################################################
for N_PARCELS in 100 200 400; do
    PARC_STR=atlas-Schaefer${N_PARCELS}7
    echo ${PARC_STR}
    ANNOT_FILE=${ATLAS_DIR}/Schaefer/${PARC_STR}/${PARC_STR}_space-fsLR_den-32k_dseg.dlabel.nii

    # myelin
    IN_FILE=${S1200_DIR}/${SUBJECT_ID}/MNINonLinear/fsaverage_LR32k/${SUBJECT_ID}.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii
    OUT_FILE=${MYELIN_OUTDIR}/${SUBJECT_ID}_${PARC_STR}_stat-mean_myelin.pscalar.nii
    wb_command -cifti-parcellate ${IN_FILE} ${ANNOT_FILE} 2 ${OUT_FILE}

    # resting-state fmri
    for SCAN in rfMRI_REST1_LR rfMRI_REST2_LR rfMRI_REST1_RL rfMRI_REST2_RL; do
        echo ${SCAN}
        SCAN_LABEL=${SCAN//_/}
        IN_FILE=${S1200_DIR}/${SUBJECT_ID}/MNINonLinear/Results/${SCAN}/${SCAN}_Atlas_MSMAll_hp2000_clean.dtseries.nii
        OUT_FILE=${FMRI_OUTDIR}/${SUBJECT_ID}_task-${SCAN_LABEL}_${PARC_STR}_stat-mean_timeseries.ptseries.nii
        wb_command -cifti-parcellate ${IN_FILE} ${ANNOT_FILE} 2 ${OUT_FILE}
    done

    # task fmri
    for SCAN in tfMRI_EMOTION_LR tfMRI_GAMBLING_LR tfMRI_LANGUAGE_LR tfMRI_MOTOR_LR tfMRI_RELATIONAL_LR tfMRI_SOCIAL_LR tfMRI_WM_LR tfMRI_EMOTION_RL tfMRI_GAMBLING_RL tfMRI_LANGUAGE_RL tfMRI_MOTOR_RL tfMRI_RELATIONAL_RL tfMRI_SOCIAL_RL tfMRI_WM_RL; do
        echo ${SCAN}
        SCAN_LABEL=${SCAN//_/}
        IN_FILE=${S1200_DIR}/${SUBJECT_ID}/MNINonLinear/Results/${SCAN}/${SCAN}_Atlas_MSMAll.dtseries.nii
        OUT_FILE=${FMRI_OUTDIR}/${SUBJECT_ID}_task-${SCAN_LABEL}_${PARC_STR}_stat-mean_timeseries.ptseries.nii
        wb_command -cifti-parcellate ${IN_FILE} ${ANNOT_FILE} 2 ${OUT_FILE}
    done

    # task fmri contrasts
    for SCAN in tfMRI_EMOTION tfMRI_GAMBLING tfMRI_LANGUAGE tfMRI_MOTOR tfMRI_RELATIONAL tfMRI_SOCIAL tfMRI_WM; do
        echo ${SCAN}
        SCAN_LABEL=${SCAN//_/}
        IN_FILE=${S1200_DIR}/${SUBJECT_ID}/MNINonLinear/Results/${SCAN}/${SCAN}_hp200_s2_level2_MSMAll.feat/${SUBJECT_ID}_${SCAN}_level2_hp200_s2_MSMAll.dscalar.nii
        OUT_FILE=${FMRI_CON_OUTDIR}/${SUBJECT_ID}_task-${SCAN_LABEL}_${PARC_STR}_stat-mean_spm.pscalar.nii
        wb_command -cifti-parcellate ${IN_FILE} ${ANNOT_FILE} 2 ${OUT_FILE}
    done
done
####################################################################################################

echo "Done! Look at you go."
