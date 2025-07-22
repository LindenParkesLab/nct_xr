#!/bin/bash
#SBATCH --requeue                 # Return job to the queue if preempted
#SBATCH --job-name=mrtrix         # Assign a short name to your job
#SBATCH --nodes=1                 # Number of nodes you require
#SBATCH --ntasks=1                # Total # of tasks across all nodes
#SBATCH --mem=32000               # Real memory (RAM) required (MB)
#SBATCH --time=1-00:00:00         # Total run time limit (D-HH:MM:SS)
#SBATCH --output=slurm_%A_%a.out  # STDOUT output file

#SBATCH --partition=main          # Partition (job queue)
# SBATCH --partition=p_dz268_1
#SBATCH --cpus-per-task=32         # Cores per task (>1 if multithread tasks)
#SBATCH --array=1-100
# SBATCH --array=1-485
# SBATCH --array=486-969

# print info about GPU
nvidia-smi

# get CPU core count
if [[ ${USER} == "lindenmp" ]]; then
    NPROC="$(nproc --all)"
    NPROC=$((NPROC / 2))
elif [[ ${USER} == "lp756" ]]; then
    NPROC=${SLURM_CPUS_PER_TASK}
fi
echo "NRPOC:" ${NPROC}
export MRTRIX_NTHREADS=${NPROC}

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
    RESDATA_DIR=/media/lindenmp/storage_ssd_1/research_data
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
FS_DIR=${T1W_DIR}/${SUBJECT_ID}
DWI_DIR=${T1W_DIR}/Diffusion
PARC_DIR=${DERIVS_DIR}/parcellations/${SUBJECT_ID}

NTCK=10m
MRTRIX_DIR=${DERIVS_DIR}/mrtrix_${NTCK}/${SUBJECT_ID}
# rm -rf ${MRTRIX_DIR}
# mkdir -p ${MRTRIX_DIR}
cd ${MRTRIX_DIR}
####################################################################################################

####################################################################################################
# copy data and convert to .mif
date
start=$(date +%s)

# dwi
mrconvert ${DWI_DIR}/data.nii.gz dwi.mif -fslgrad ${DWI_DIR}/bvecs ${DWI_DIR}/bvals -datatype float32 -stride 0,0,0,1
mrconvert ${DWI_DIR}/nodif_brain_mask.nii.gz mask.mif

# t1w
mrconvert ${T1W_DIR}/T1w_acpc_dc_restore_brain.nii.gz T1w.mif

end=$(date +%s)
echo "Elapsed Time: $((($end-$start) / 60)) minutes"
####################################################################################################

####################################################################################################
# create dwi mask
date
start=$(date +%s)

maskfilter mask.mif dilate dwi_mask.mif -npass 5

end=$(date +%s)
echo "Elapsed Time: $((($end-$start) / 60)) minutes"

# plots
# mrview dwi.mif -roi.load dwi_mask.mif -overlay.opacity 0.6 -mode 2  # check mask
####################################################################################################

####################################################################################################
# minimal pre-processing
date
start=$(date +%s)

dwibiascorrect ants dwi.mif dwi_bc.mif -bias bias.mif -mask dwi_mask.mif
dwiextract dwi_bc.mif - -bzero | mrmath - mean mean_b0.mif -axis 3

end=$(date +%s)
echo "Elapsed Time: $((($end-$start) / 60)) minutes"
####################################################################################################

####################################################################################################
# local fibre orientation distribution estimation
date
start=$(date +%s)

dwi2response dhollander dwi_bc.mif wm.txt gm.txt csf.txt -voxels voxels.mif -mask dwi_mask.mif  # estimate response function
dwi2fod msmt_csd dwi_bc.mif -mask dwi_mask.mif \
  wm.txt wmfod.mif gm.txt gm.mif csf.txt csf.mif  # generate orientation distribution functions (ODFs)
mtnormalise wmfod.mif wmfod_norm.mif gm.mif gm_norm.mif csf.mif csf_norm.mif -mask dwi_mask.mif \
  -check_factors check_factors.txt -check_norm check_norm.mif -check_mask check_mask.mif  # Bias field correction and Intensity Normalization

end=$(date +%s)
echo "Elapsed Time: $((($end-$start) / 60)) minutes"
####################################################################################################

####################################################################################################
# Create thalamus mask
date
start=$(date +%s)

echo "Creating thalamic mask file"

# Output mask file
INPUT_FILE="${FS_DIR}/mri/aseg.hires.nii.gz"
fslmaths ${INPUT_FILE} -thr 10 -uthr 10 -bin thalamus_mask-left.nii.gz
fslmaths ${INPUT_FILE} -thr 49 -uthr 49 -bin thalamus_mask-right.nii.gz

# Combine the two masks (left + right thalamus) into a single mask
fslmaths thalamus_mask-left.nii.gz -add thalamus_mask-right.nii.gz -bin thalamus_mask.nii.gz
mrconvert thalamus_mask.nii.gz thalamus_mask.mif -force

python ${SCRIPT_DIR}/label_thalamus_voxels.py thalamus_mask.nii.gz labeled_thalamus_mask.nii.gz thalamus_voxel_coordinates.txt
mrconvert labeled_thalamus_mask.nii.gz labeled_thalamus_mask.mif -force

end=$(date +%s)
echo "Elapsed Time: $((($end-$start) / 60)) minutes"
####################################################################################################

####################################################################################################
# Generate whole-brain tractogram
date
start=$(date +%s)

5ttgen hsvs ${FS_DIR} 5tt.mif  # create 5 tissue type image

# 1) whole-brain seeding
# tckgen -algorithm ifod2 \
#   -act 5tt.mif -backtrack -seed_dynamic wmfod_norm.mif \
#   -select ${NTCK} wmfod_norm.mif tracks.tck  # generate streamlines
# tcksift2 -act 5tt.mif \
#   tracks.tck wmfod_norm.mif \
#   sift2_weights.txt -out_mu sift2_mu.txt  # optimize tractogram
# tckedit tracks.tck -number 200k tracks_200k.tck  # downsample to 200k streamlines (just for visualization)

# 2) thalamic seeding
tckgen -algorithm ifod2 \
  -act 5tt.mif -backtrack -seed_image thalamus_mask.mif \
  -select 1m wmfod_norm.mif tracks_thalamus_to_brain_1m.tck
tcksift2 -act 5tt.mif \
  tracks_thalamus_to_brain_1m.tck wmfod_norm.mif \
  sift2_weights_thalamus_1m.txt -out_mu sift2_mu_thalamus_1m.txt

end=$(date +%s)
echo "Elapsed Time: $((($end-$start) / 60)) minutes"
####################################################################################################

####################################################################################################
# Generate structural connectivity matrix
date
start=$(date +%s)

# 1) whole-brain connectome
# for PARC_FILE in ${PARC_DIR}/*_space-T1w_res-1_dseg.nii.gz; do
#   mrconvert ${PARC_FILE} parc.mif -datatype uint32

#   OUT_FILE=$(basename -- ${PARC_FILE})
#   OUT_FILE=${OUT_FILE%.nii.gz}
#   OUT_FILE=$(echo ${OUT_FILE} | cut -d "_" -f 1,2,3)
#   OUT_FILE=${OUT_FILE}_stat-streamlinecount_relmat.csv
#   echo ${OUT_FILE}

#   tck2connectome -tck_weights_in sift2_weights.txt -symmetric -zero_diagonal tracks.tck parc.mif ${OUT_FILE}
#   rm -f parc.mif
# done

# 2) thalamo-cortical connectome
for N_PARCELS in 100 200 400; do
  PARC_FILE=${PARC_DIR}/${SUBJECT_ID}_atlas-Schaefer${N_PARCELS}7_space-T1w_res-1_dseg.nii.gz
  TSV_FILE=${ATLAS_DIR}/Schaefer/atlas-Schaefer${N_PARCELS}7/atlas-Schaefer${N_PARCELS}7_dseg.tsv

  mrconvert ${PARC_FILE} parc.mif -datatype uint32

  OUT_FILE=$(basename -- ${PARC_FILE})
  OUT_FILE=${OUT_FILE%.nii.gz}
  OUT_FILE=$(echo ${OUT_FILE} | cut -d "_" -f 1,2,3)
  OUT_FILE=${OUT_FILE}_stat-streamlinecount_relmatthal.csv
  echo ${OUT_FILE}

  mrcalc labeled_thalamus_mask.mif 0 -gt labeled_thalamus_mask.mif parc.mif -if parc_with_thal.mif -datatype uint32
  tck2connectome -tck_weights_in sift2_weights_thalamus_1m.txt -symmetric -zero_diagonal tracks_thalamus_to_brain_1m.tck parc_with_thal.mif ${OUT_FILE}

  python3 ${SCRIPT_DIR}/filter_connectome.py ${OUT_FILE} ${OUT_FILE} ${TSV_FILE}
  OUT_FILE_NII=${OUT_FILE%.csv}
  OUT_FILE_NII=${OUT_FILE_NII}.nii.gz
  python3 ${SCRIPT_DIR}/create_nii_from_connectome.py ${OUT_FILE} ${OUT_FILE_NII} thalamus_voxel_coordinates.txt labeled_thalamus_mask.nii.gz

  rm -f parc.mif parc_with_thal.mif
done

end=$(date +%s)
echo "Elapsed Time: $((($end-$start) / 60)) minutes"
####################################################################################################

####################################################################################################
# clean up some files
rm -f 5tt.mif bias.mif check_mask.mif check_norm.mif csf.mif csf_norm.mif dwi_bc.mif dwi_mask.mif \
  dwi.mif gm.mif gm_norm.mif labeled_thalamus_mask.mif mask.mif mean_b0.mif T1w.mif thalamus_mask.mif voxels.mif wmfod.mif wmfod_norm.mif
rm -f thalamus_mask-left.nii.gz thalamus_mask-right.nii.gz labeled_thalamus_mask.nii.gz
rm -f check_factors.txt csf.txt diff2struct_mrtrix.txt gm.txt sift2_mu*.txt wm.txt thalamus_voxel_coordinates.txt
####################################################################################################

echo "Done! Look at you go."
