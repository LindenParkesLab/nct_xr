#!/bin/bash
#SBATCH --requeue                 # Return job to the queue if preempted
#SBATCH --job-name=parcgen        # Assign a short name to your job
#SBATCH --nodes=1                 # Number of nodes you require
#SBATCH --ntasks=1                # Total # of tasks across all nodes
#SBATCH --mem=8000                # Real memory (RAM) required (MB)
#SBATCH --time=02:00:00           # Total run time limit (D-HH:MM:SS)
#SBATCH --output=slurm_%A_%a.out  # STDOUT output file

#SBATCH --partition=main          # Partition (job queue)
# SBATCH --partition=p_dz268_1
#SBATCH --cpus-per-task=12         # Cores per task (>1 if multithread tasks)
#SBATCH --array=1-485
# SBATCH --array=486-969

# print info about GPU
nvidia-smi

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
    SCRIPT_DIR=/home/lindenmp/research_projects/connectome_generator/scripts
elif [[ ${USER} == "lp756" ]]; then
    RESDATA_DIR=/scratch/f_ah1491_1/open_data
    export FREESURFER_HOME=${HOME}/freesurfer
    TEMPLATEFLOW_HOME=${RESDATA_DIR}/templateflow
    ATLAS_DIR=${RESDATA_DIR}/atlases
    SCRIPT_DIR=/scratch/f_ah1491_1/open_data/HCP_YA/scripts
fi

PROJ_DIR=${RESDATA_DIR}/HCP_YA
S1200_DIR=${PROJ_DIR}/HCP_1200
DERIVS_DIR=${PROJ_DIR}/derivatives
####################################################################################################

####################################################################################################
# subject variables
SUBJECT_ID=$(sed -n ${IDX}p ${PROJ_DIR}/HCPYA_Schaefer4007_subjids.txt)
echo ${SUBJECT_ID}

T1W_DIR=${S1200_DIR}/${SUBJECT_ID}/T1w
PARC_DIR=${DERIVS_DIR}/parcellations/${SUBJECT_ID}
rm -rf ${PARC_DIR}
mkdir -p ${PARC_DIR}
cd ${PARC_DIR}
####################################################################################################

####################################################################################################
# run ants
date
start=$(date +%s)

SPACE=MNI152NLin2009cAsym
# SPACE=MNI152NLin6Asym
MNI_TEMPLATE=${TEMPLATEFLOW_HOME}/tpl-${SPACE}/tpl-${SPACE}_res-01_desc-brain_T1w.nii.gz
IN_FILE=${T1W_DIR}/T1w_acpc_dc_restore_brain.nii.gz
mrconvert ${IN_FILE} T1w_brain.nii
antsRegistrationSyN.sh -d 3 -f ${MNI_TEMPLATE} -m T1w_brain.nii -o T1w_MNI_ -n ${NPROC}

end=$(date +%s)
echo "Elapsed Time: $((($end-$start) / 60)) minutes"
####################################################################################################

####################################################################################################
# generate subject specific parcellations
date
start=$(date +%s)

# schaefer
for N_PARCELS in 100 200 400; do
    PARC_STR=atlas-Schaefer${N_PARCELS}7
    IN_FILE=${ATLAS_DIR}/Schaefer/${PARC_STR}/${PARC_STR}_space-${SPACE}_res-1_dseg.nii.gz
    OUT_FILE=${SUBJECT_ID}_${PARC_STR}_space-T1w_res-1_dseg.nii.gz
    antsApplyTransforms -d 3 -i ${IN_FILE} -r T1w_brain.nii -o ${OUT_FILE} -t [T1w_MNI_0GenericAffine.mat, 1] -t T1w_MNI_1InverseWarp.nii.gz -n NearestNeighbor
    python ${SCRIPT_DIR}/get_region_sizes.py ${OUT_FILE}
done

# schaefer + subcortex + cerebellum
for N_PARCELS in 156 256 456; do
    PARC_STR=atlas-4S${N_PARCELS}Parcels
    IN_FILE=${ATLAS_DIR}/4SParcels/${PARC_STR}/${PARC_STR}_space-${SPACE}_res-1_dseg.nii.gz
    OUT_FILE=${SUBJECT_ID}_${PARC_STR}_space-T1w_res-1_dseg.nii.gz
    antsApplyTransforms -d 3 -i ${IN_FILE} -r T1w_brain.nii -o ${OUT_FILE} -t [T1w_MNI_0GenericAffine.mat, 1] -t T1w_MNI_1InverseWarp.nii.gz -n NearestNeighbor
    python ${SCRIPT_DIR}/get_region_sizes.py ${OUT_FILE}
done

# schaefer + melbourne subcortex + cerebellum
# for N_PARCELS in 100 200 400; do
#     PARC_STR=atlas-Schaefer${N_PARCELS}7
#     IN_FILE=${ATLAS_DIR}/Schaefer/${PARC_STR}/${PARC_STR}_space-${SPACE}_res-1_dseg.nii.gz
#     OUT_FILE=${SUBJECT_ID}_${PARC_STR}_space-T1w_res-1_dseg.nii.gz
#     antsApplyTransforms -d 3 -i ${IN_FILE} -r T1w_brain.nii -o ${OUT_FILE} -t [T1w_MNI_0GenericAffine.mat, 1] -t T1w_MNI_1InverseWarp.nii.gz -n NearestNeighbor

#     for SCALE in 1 2 3 4; do
#         PARC_STR=atlas-Schaefer${N_PARCELS}7MSA${SCALE}MDTB10
#         IN_FILE=${ATLAS_DIR}/SchaeferMSAMDTB10/${PARC_STR}/${PARC_STR}_space-${SPACE}_res-1_dseg.nii.gz
#         OUT_FILE=${SUBJECT_ID}_${PARC_STR}_space-T1w_res-1_dseg.nii.gz
#         antsApplyTransforms -d 3 -i ${IN_FILE} -r T1w_brain.nii -o ${OUT_FILE} -t [T1w_MNI_0GenericAffine.mat, 1] -t T1w_MNI_1InverseWarp.nii.gz -n NearestNeighbor
#     done
# done

# glasser
# PARC_STR=atlas-Glasser
# IN_FILE=${ATLAS_DIR}/Glasser/${PARC_STR}/${PARC_STR}_space-${SPACE}_res-1_dseg.nii.gz
# OUT_FILE=${SUBJECT_ID}_${PARC_STR}_space-T1w_res-1_dseg.nii.gz
# antsApplyTransforms -d 3 -i ${IN_FILE} -r T1w_brain.nii -o ${OUT_FILE} -t [T1w_MNI_0GenericAffine.mat, 1] -t T1w_MNI_1InverseWarp.nii.gz -n NearestNeighbor

end=$(date +%s)
echo "Elapsed Time: $((($end-$start) / 60)) minutes"

# plots
# mrview ${MNI_TEMPLATE} -overlay.load T1w_MNI_Warped.nii.gz -overlay.opacity 0.6 -mode 2  # check mask
# mrview T1w.nii -mode 2 -roi.load Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S4_T1w.nii.gz
####################################################################################################

echo "Done! Look at you go."
