#!/bin/bash
#SBATCH --requeue                 # Return job to the queue if preempted
#SBATCH --job-name=nct_xr         # Assign a short name to your job
#SBATCH --nodes=1                 # Number of nodes you require
#SBATCH --ntasks=1                # Total # of tasks across all nodes
#SBATCH --mem=12000               # Real memory (RAM) required (MB)
#SBATCH --time=3-00:00:00         # Total run time limit (D-HH:MM:SS)
#SBATCH --output=slurm_%A_%a.out  # STDOUT output file

# for gpu
# SBATCH --partition=gpu           # Partition (job queue)
# SBATCH --cpus-per-task=2         # Cores per task (>1 if multithread tasks)
# SBATCH --gres=gpu:1              # use GPUs
# SBATCH --array=0-199
# SBATCH --constraint=adalovelace|ampere|titan|volta|pascal
# SBATCH --constraint=adalovelace|ampere

# for cpu
#SBATCH --partition=main          # Partition (job queue)
# SBATCH --partition=p_dz268_1     # Partition (job queue)
#SBATCH --cpus-per-task=10        # Cores per task (>1 if multithread tasks)
#SBATCH --array=0-499

# Print the task id.
echo "SLURM_ARRAY_TASK_ID:" ${SLURM_ARRAY_TASK_ID}

# print info about GPU
nvidia-smi

########################################################################################################################
# directories
projdir='/projectsp/f_lp756_1/lindenmp/research_projects/nct_xr'
scriptsdir=${projdir}'/scripts'
indir=${projdir}'/data/int_deflections'

########################################################################################################################
time_horizon=1
reference_state='xf'
########################################################################################################################

########################################################################################################################
# subject-level connectomes: hcp-ya
outdir=${projdir}'/results/int_deflections/HCPYA'
outsubdir='subjects'
A_file=${indir}'/HCPYA_Schaefer4007_A.npy'
fmri_clusters_file=${outdir}'/HCPYA_Schaefer4007_states.npy'
file_prefix='HCPYA-Schaefer4007'

for i in $(seq 0 500 1000); do
    perm_idx=$((${SLURM_ARRAY_TASK_ID} + ${i}))
    echo "perm_idx:" ${perm_idx}

    python ${scriptsdir}/compute_optimized_control_energy.py --A_file ${A_file} --fmri_clusters_file ${fmri_clusters_file} \
        --outdir ${outdir} --outsubdir ${outsubdir} --file_prefix ${file_prefix} \
        --time_horizon ${time_horizon} --reference_state ${reference_state} \
        --perm_idx ${perm_idx} \
        --compact_save 'True'
done
########################################################################################################################

